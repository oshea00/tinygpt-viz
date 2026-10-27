// Plays a lesson: applies each step to the shared state, and checks answers live.

import { concepts } from "../concepts/store.svelte";
import { app, selection, settings } from "../state/app.svelte";
import { available, isCorrect, lessonVars, loadLessons, type Lesson, type LessonStep } from "./lesson";

class Player {
  readonly lessons = loadLessons();
  lesson = $state.raw<Lesson | null>(null);
  index = $state(0);
  menuOpen = $state(false);
  revealed = $state(false);     // the answer to a "reveal" question was shown

  /** Whether the server allows training (started with --train). */
  get training(): boolean {
    return app.hello?.training ?? false;
  }

  get steps(): LessonStep[] {
    return this.lesson?.steps ?? [];
  }

  get step(): LessonStep | null {
    return this.steps[this.index] ?? null;
  }

  get vars(): Record<string, string> {
    return app.hello ? lessonVars(app.hello.config) : {};
  }

  /** Whether the current step's question is answered by the current selection. */
  get answered(): boolean {
    const ask = this.step?.ask;
    if (!ask || !app.hello) return false;
    return isCorrect(ask.check, {
      stageId: selection.stageId, position: selection.position, dim: selection.dim,
      layer: selection.layer, head: selection.head, tab: settings.tab,
    }, app.trace, app.hello.config.n_layer, app.hello.config.n_head, {
      lensSnap: settings.lensSnap, lens: app.serverLens, revealed: this.revealed,
      captured: !!concepts.current && concepts.pending === 0 && (!concepts.withBaseline || !!concepts.untrained),
      trained: !!app.stepExplain,
      patches: concepts.patches,
      patchCell: app.interventions.some((iv) => iv.kind === "patch") ? concepts.patchCell : null,
    });
  }

  start(lesson: Lesson): void {
    if (!available(lesson, this.training)) return;
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
    this.index = Math.max(0, Math.min(this.steps.length - 1, i));
    this.revealed = false;
    if (this.apply(this.step!)) app.run();
  }

  /** Back to how the current step began: a step only sets what it changes and inherits the
   * rest (prompt, tab, stage) from the steps before it, so replay them all from a clean
   * selection with no interventions. Answers are worked out from the selection and from what
   * the question asked for (a capture, a training step), so clearing those un-answers it. */
  reset(): void {
    if (!this.lesson) return;
    app.stopAuto();
    selection.reset();
    settings.lensSnap = -1;
    this.revealed = false;
    const check = this.step?.ask?.check.kind;
    if (check === "captured") concepts.current = concepts.untrained = null;
    if (check === "trained") app.stepExplain = null;
    let rerun = app.interventions.length > 0;
    app.interventions = [];
    for (const s of this.steps.slice(0, this.index + 1)) rerun = this.apply(s) || rerun;
    if (rerun) app.run();
  }

  /** Apply one step's settings; true when the model must run again to show them. */
  private apply(s: LessonStep): boolean {
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
    if (s.kvCache !== undefined && s.kvCache !== app.kvCache) {
      app.kvCache = s.kvCache;
      rerun = true;
    }
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
    return rerun;
  }
}

export const player = new Player();
