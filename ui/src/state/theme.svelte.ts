// Light or dark: the system's preference until the user picks one, which is remembered.
// app.css defines both palettes; data-theme on <html> overrides the system preference.

type Theme = "light" | "dark";

const KEY = "tinygpt-viz.theme";
const media = window.matchMedia("(prefers-color-scheme: dark)");

function stored(): Theme | null {
  try {
    const v = localStorage.getItem(KEY);
    return v === "light" || v === "dark" ? v : null;
  } catch {
    return null;   // storage blocked: follow the system
  }
}

class ThemeState {
  chosen = $state<Theme | null>(stored());
  system = $state<Theme>(media.matches ? "dark" : "light");

  /** The theme in effect. Canvas views read it so they repaint when it changes. */
  get current(): Theme {
    return this.chosen ?? this.system;
  }

  toggle(): void {
    this.chosen = this.current === "dark" ? "light" : "dark";
    try {
      localStorage.setItem(KEY, this.chosen);
    } catch {
      // not remembered, but still applied
    }
    this.apply();
  }

  apply(): void {
    if (this.chosen) document.documentElement.dataset.theme = this.chosen;
    else delete document.documentElement.dataset.theme;
  }
}

export const theme = new ThemeState();
media.addEventListener("change", (e) => (theme.system = e.matches ? "dark" : "light"));
theme.apply();   // on import, before the first render, so the page never flashes the other theme
