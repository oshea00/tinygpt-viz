import { mount } from "svelte";
import "katex/dist/katex.min.css";
import "./app.css";
import App from "./App.svelte";
import { app } from "./state/app.svelte";
import "./state/theme.svelte";   // applies the saved theme before the first render

app.start();
export default mount(App, { target: document.getElementById("app")! });
