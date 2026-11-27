import { defineStore } from "pinia";
import { ref } from "vue";
import { loadString, saveString } from "@/lib/storage";

/** The pawn-server bearer token (api.token), kept in localStorage. */
export const useAuthStore = defineStore("auth", () => {
  const token = ref(loadString("token"));
  /** True when the token is missing or the server rejected it (401). */
  const needsToken = ref(!token.value);

  function setToken(value: string): void {
    token.value = value.trim();
    saveString("token", token.value || null);
    needsToken.value = !token.value;
  }

  function signOut(): void {
    setToken("");
  }

  function rejected(): void {
    needsToken.value = true;
  }

  return { token, needsToken, setToken, signOut, rejected };
});
