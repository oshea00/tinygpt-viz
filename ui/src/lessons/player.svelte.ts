// Plays a lesson: applies each step to the shared state, and checks answers live.

import { concepts } from "../concepts/store.svelte";
import { app, selection, settings } from "../state/app.svelte";
import { isCorrect, loadLessons, type Lesson, type LessonStep } from "./lesson";

class Player {
  readonly lessons = loadLessons();
  lesson = $state.raw<Lesson | null>(null);
  index = $state(0);
  menuOpen = $state(false);

  get step(): LessonStep | null {
    return this.lesson?.steps[this.index] ?? null;
  }

  /** Whether the current step's question is answered by the current selection. */
  get answered(): boolean {
    const ask = this.step?.ask;
    if (!ask || !app.hello) return false;
    return isCorrect(ask.check, {
      stageId: selection.stageId, position: selection.position, dim: selection.dim,
      layer: selection.layer, head: selection.head, tab: settings.tab,
    }, app.trace, app.hello.config.n_layer, app.hello.config.n_head);
  }

  start(lesson: Lesson): void {
    this.lesson = lesson;
    this.menuOpen = false;
    app.stopAuto();
    if (app.interventions.length) app.setInterventions([]);
    this.go(0);
  }

  stop(): void {
    this.lesson = null;
  }

  go(i: number): void {
    if (!this.lesson) return;
    this.index = Math.max(0, Math.min(this.lesson.steps.length - 1, i));
    this.apply(this.lesson.steps[this.index]);
  }

  private apply(s: LessonStep): void {
    let rerun = false;
    if (s.prompt !== undefined && s.prompt !== app.prompt) {
      app.prompt = s.prompt;
      app.promptEdited();
      rerun = true;
    }
    if (s.interventions) {
      app.interventions = s.interventions;
      rerun = true;
    }
    if (s.sampling) app.sampling = { temperature: s.sampling.temperature, topK: s.sampling.topK ?? null, topP: s.sampling.topP ?? null };
    if (s.tab) settings.tab = s.tab;
    if (s.head) selection.selectHead(s.head[0], s.head[1]);
    if (s.stage) selection.select(s.stage, s.phase ?? "forward");
    else if (s.phase) selection.phase = s.phase;
    if (s.position !== undefined) selection.position = s.position === "last" ? -1 : s.position;
    if (s.dim !== undefined) selection.dim = s.dim;
    if (s.atlasMode) settings.atlasMode = s.atlasMode;
    if (s.lensMode) settings.lensMode = s.lensMode;
    if (s.residualMode) settings.residualMode = s.residualMode;
    if (s.conceptPanel) concepts.panel = s.conceptPanel;
    settings.compare = false;
    if (rerun) app.run();
  }
}

export const player = new Player();
