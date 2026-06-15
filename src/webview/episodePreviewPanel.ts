// Episode preview webview host.
// Owns one panel per (dataset, episode) pair; opening the same pair again
// just reveals the existing panel.

import * as vscode from "vscode";
import { resolveVideoUri } from "../dataset/datasetLoader";
import type { DatasetService } from "../dataset/datasetService";
import { readEpisodeSignals } from "../dataset/parquetReader";
import { log, logError } from "../log";
import { launchRerun } from "../rerun/rerunLauncher";
import { serveVideo, stopServeVideo } from "./videoServer";
import type { DatasetMetadataView, LeRobotEpisode } from "../types";
import { BaseWebviewPanel } from "./baseWebviewPanel";
import type { FromWebviewMessage } from "./protocol";

export class EpisodePreviewPanelManager implements vscode.Disposable {
  private panel: EpisodePreviewPanel | undefined;

  constructor(
    private readonly context: vscode.ExtensionContext,
    private readonly service: DatasetService,
  ) {}

  async show(datasetId: string, episodeIndex: number): Promise<void> {
    let snapshot;
    try {
      snapshot = await this.service.getSnapshot(datasetId);
    } catch (err) {
      logError(`opening episode ${episodeIndex} of ${datasetId}`, err);
      vscode.window.showErrorMessage(`Could not load dataset: ${(err as Error).message}`);
      return;
    }

    const episode = snapshot.episodes.find((e) => e.episodeIndex === episodeIndex);
    if (!episode) {
      vscode.window.showErrorMessage(
        `Episode ${episodeIndex} not found in ${snapshot.descriptor.name}`,
      );
      return;
    }

    if (this.panel) {
      this.panel.setEpisode(snapshot.descriptor, episode);
      this.panel.reveal();
    } else {
      this.panel = new EpisodePreviewPanel(
        this.context, this.service, snapshot.descriptor, episode,
      );
      this.panel.onDidDispose(() => { this.panel = undefined; });
    }
  }

  /** Show dataset metadata in the shared preview panel. */
  async showMetadata(datasetId: string): Promise<void> {
    let snapshot;
    try {
      snapshot = await this.service.getSnapshot(datasetId);
    } catch (err) {
      logError(`opening metadata for ${datasetId}`, err);
      vscode.window.showErrorMessage(`Could not load dataset: ${(err as Error).message}`);
      return;
    }
    const metadata = buildMetadataView(snapshot);

    if (this.panel) {
      this.panel.setDataset(snapshot.descriptor);
      this.panel.post({ type: "init-metadata", data: metadata });
      this.panel.reveal();
    } else {
      const ep = snapshot.episodes[0];
      if (!ep) {
        vscode.window.showErrorMessage("Dataset has no episodes to preview.");
        return;
      }
      this.panel = new EpisodePreviewPanel(
        this.context, this.service, snapshot.descriptor, ep, true, // metadataMode
      );
      this.panel.onDidDispose(() => { this.panel = undefined; });
      this.panel.post({ type: "init-metadata", data: metadata });
    }
  }

  dispose(): void {
    this.panel?.dispose();
    this.panel = undefined;
  }
}

class EpisodePreviewPanel extends BaseWebviewPanel {
  constructor(
    context: vscode.ExtensionContext,
    private readonly service: DatasetService,
    descriptor: { id: string; name: string; root?: string },
    private episode: LeRobotEpisode,
    private _metadataMode = false,
  ) {
    super({
      context,
      viewType: "lerobotEpisodePreview",
      title: `${descriptor.name} · Episode ${episode.episodeIndex}`,
      extraResourceRoots: descriptor.root ? [vscode.Uri.file(descriptor.root)] : [],
    });
    this.datasetId = descriptor.id;
    this.descriptor = descriptor;
  }

  private datasetId: string;
  private descriptor: { id: string; name: string; root?: string };
  private _initialized = false;
  /** Video file paths currently served via HTTP servers — stopped on switch. */
  private _servedVideoPaths: string[] = [];


  /** Update the dataset descriptor without changing the episode (for metadata view). */
  setDataset(descriptor: { id: string; name: string; root?: string }): void {
    const datasetChanged = this.descriptor?.root !== descriptor.root;
    this.datasetId = descriptor.id;
    this.descriptor = descriptor;
    this.panel.title = `${descriptor.name}`;
    if (datasetChanged && descriptor.root) {
      this.panel.webview.options = {
        enableScripts: true,
        localResourceRoots: [
          vscode.Uri.joinPath(this.context.extensionUri, "dist"),
          vscode.Uri.joinPath(this.context.extensionUri, "media"),
          vscode.Uri.file(descriptor.root),
        ],
      };
    }
  }

  /** Update the panel to show a different episode (same or different dataset). */
  setEpisode(descriptor: { id: string; name: string; root?: string }, episode: LeRobotEpisode): void {
    this._metadataMode = false;
    const datasetChanged = this.descriptor?.root !== descriptor.root;
    this.datasetId = descriptor.id;
    this.descriptor = descriptor;
    this.episode = episode;
    this.panel.title = `${descriptor.name} · Episode ${episode.episodeIndex}`;

    // When switching to a different dataset, update localResourceRoots so
    // the webview can access video files from the new dataset's directory.
    if (datasetChanged && descriptor.root) {
      this.panel.webview.options = {
        enableScripts: true,
        localResourceRoots: [
          vscode.Uri.joinPath(this.context.extensionUri, "dist"),
          vscode.Uri.joinPath(this.context.extensionUri, "media"),
          vscode.Uri.file(descriptor.root),
        ],
      };
    }

    // Trigger the webview to reload with new episode data.
    void this.refreshPreview();
  }

  private async refreshPreview(): Promise<void> {
    try {
      const snapshot = await this.service.getSnapshot(this.datasetId);
      const meta = await this.buildMeta(snapshot);
      this.post({ type: "init-meta", data: meta });
      const signals = await this.buildSignals(snapshot);
      this.post({ type: "init-signals", data: signals });
    } catch (err) {
      logError("refreshPreview", err);
    }
  }

  override dispose(): void {
    for (const p of this._servedVideoPaths) stopServeVideo(p);
    this._servedVideoPaths = [];
    super.dispose();
  }

  protected override extraCspDirectives(): Array<[string, string]> {
    return [["media-src", `${this.panel.webview.cspSource} https: blob: http://127.0.0.1:*`]];
  }

  protected async onMessage(message: FromWebviewMessage): Promise<void> {
    switch (message.type) {
      case "ready": {
        if (!this._initialized) {
          this._initialized = true;
          if (!this._metadataMode) await this.refreshPreview();
        }
        return;
      }
      case "open-in-rerun": {
        const snapshot = await this.service.getSnapshot(this.datasetId);
        await launchRerun(snapshot, this.episode);
        return;
      }
      case "open-source-file": {
        await vscode.commands.executeCommand("vscode.open", vscode.Uri.file(message.path));
        return;
      }
      case "frame-changed":
        return;
    }
  }

  private async buildMeta(snapshot: Awaited<ReturnType<DatasetService["getSnapshot"]>>) {
    // Stop video servers from the previous episode to free ports.
    for (const p of this._servedVideoPaths) stopServeVideo(p);
    this._servedVideoPaths = [];

    const cameras = await Promise.all(
      snapshot.cameraKeys.map(async (key) => {
        const resolved = await resolveVideoUri(snapshot, this.episode, key);
        if (!resolved) return { key };
        // Large videos → stream via our own HTTP server (supports Range).
        // Small videos → VS Code webview local server (simpler, no Range needed).
        let sz = 0;
        try { sz = (await import("node:fs/promises").then((m) => m.stat(resolved.uri.fsPath))).size; } catch { /* ok */ }
        const useStreaming = sz > 50 * 1024 * 1024;
        let videoUri: string;
        if (useStreaming) {
          try {
            videoUri = await serveVideo(resolved.uri.fsPath);
            this._servedVideoPaths.push(resolved.uri.fsPath);
          } catch {
            videoUri = this.panel.webview.asWebviewUri(resolved.uri).toString();
          }
        } else {
          videoUri = this.panel.webview.asWebviewUri(resolved.uri).toString();
        }
        return {
          key,
          videoUri,
          shardFrameRange: resolved.location.shardFrameRange,
          note: resolved.location.note,
        };
      }),
    );
    const rerunEnabled =
      vscode.workspace.getConfiguration("lerobotViewer").get<boolean>("enableRerun") ?? false;
    const episodeSplit = pickSplitForEpisode(snapshot.splits, this.episode.episodeIndex);

    return {
      dataset: snapshot.descriptor,
      version: snapshot.version,
      info: snapshot.info,
      episode: this.episode,
      cameras,
      stateNames: collectFeatureNames(snapshot.info.features, snapshot.stateKeys),
      actionNames: collectFeatureNames(snapshot.info.features, snapshot.actionKeys),
      velocityNames: collectFeatureNames(snapshot.info.features, snapshot.velocityKeys),
      effortNames: collectFeatureNames(snapshot.info.features, snapshot.effortKeys),
      environmentStateNames: collectFeatureNames(
        snapshot.info.features,
        snapshot.environmentStateKeys,
      ),
      rerunEnabled,
      tasks: snapshot.tasks,
      episodeLengths: snapshot.episodes.map((e) => e.length || 0),
      totalEpisodes: snapshot.info.totalEpisodes,
      stats: snapshot.stats,
      splits: snapshot.splits,
      episodeSplit,
    };
  }

  private async buildSignals(snapshot: Awaited<ReturnType<DatasetService["getSnapshot"]>>) {
    const signals = await readEpisodeSignals(snapshot, this.episode);
    log(
      `Preview signals for ${snapshot.descriptor.name} ep ${this.episode.episodeIndex}: ` +
        `state=${signals.state?.length ?? 0}f, action=${signals.action?.length ?? 0}f`,
    );
    return {
      state: signals.state,
      action: signals.action,
      velocity: signals.velocity,
      effort: signals.effort,
      environmentState: signals.environmentState,
      reward: signals.reward,
      done: signals.done,
      success: signals.success,
      truncated: signals.truncated,
      taskIndices: signals.taskIndices,
      signalsWarning: signals.warning,
    };
  }
}

function collectFeatureNames(
  features: Record<string, { names?: unknown }>,
  keys: string[],
): string[] | undefined {
  const collected: string[] = [];
  for (const key of keys) {
    const f = features[key];
    if (!f) continue;
    if (Array.isArray(f.names)) {
      collected.push(...(f.names as string[]));
    } else if (f.names && typeof f.names === "object") {
      // Some datasets nest names under a sub-key like {"motors": [...]}.
      for (const v of Object.values(f.names as Record<string, unknown>)) {
        if (Array.isArray(v)) collected.push(...(v as string[]));
      }
    }
  }
  return collected.length > 0 ? collected : undefined;
}

function pickSplitForEpisode(
  splits: Record<string, [number, number]>,
  episodeIndex: number,
): string | undefined {
  for (const [name, [from, to]] of Object.entries(splits)) {
    if (episodeIndex >= from && episodeIndex < to) return name;
  }
  return undefined;
}

function buildMetadataView(
  snapshot: Awaited<ReturnType<DatasetService["getSnapshot"]>>,
): DatasetMetadataView {
  return {
    descriptor: snapshot.descriptor,
    version: snapshot.version,
    info: snapshot.info,
    cameraKeys: snapshot.cameraKeys,
    stateKeys: snapshot.stateKeys,
    actionKeys: snapshot.actionKeys,
    velocityKeys: snapshot.velocityKeys,
    effortKeys: snapshot.effortKeys,
    environmentStateKeys: snapshot.environmentStateKeys,
    rewardKey: snapshot.rewardKey,
    doneKey: snapshot.doneKey,
    successKey: snapshot.successKey,
    truncatedKey: snapshot.truncatedKey,
    taskIndexKey: snapshot.taskIndexKey,
    tasks: snapshot.tasks,
    stats: snapshot.stats,
    splits: snapshot.splits,
    episodeLengths: snapshot.episodes.map((e) => e.length || 0),
    warnings: snapshot.warnings,
  };
}
