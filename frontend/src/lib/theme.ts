import { useEffect, useState } from "react";

export type Theme = "light" | "dark";
const THEME_KEY = "statagent.theme";

function systemTheme(): Theme {
  return typeof window !== "undefined" && window.matchMedia?.("(prefers-color-scheme: dark)").matches
    ? "dark"
    : "light";
}

function storedTheme(): Theme | null {
  try {
    const value = localStorage.getItem(THEME_KEY);
    return value === "light" || value === "dark" ? value : null;
  } catch {
    return null;
  }
}

/** Theme follows the OS until the user toggles it; the choice is remembered. */
export function useTheme(): [Theme, () => void] {
  const [theme, setTheme] = useState<Theme>(() => storedTheme() ?? systemTheme());

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
  }, [theme]);

  useEffect(() => {
    if (storedTheme()) return;
    const media = window.matchMedia?.("(prefers-color-scheme: dark)");
    const onChange = () => setTheme(systemTheme());
    media?.addEventListener("change", onChange);
    return () => media?.removeEventListener("change", onChange);
  }, []);

  const toggle = () =>
    setTheme((current) => {
      const next = current === "dark" ? "light" : "dark";
      try {
        localStorage.setItem(THEME_KEY, next);
      } catch {
        // ignore unavailable storage
      }
      return next;
    });

  return [theme, toggle];
}

/** Okabe-Ito: a colorblind-safe categorical palette. */
export const COLORWAY = [
  "#0072B2",
  "#E69F00",
  "#009E73",
  "#CC79A7",
  "#56B4E9",
  "#D55E00",
  "#F0E442",
  "#000000",
];

export function prefersReducedMotion(): boolean {
  return typeof window !== "undefined" && !!window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;
}
