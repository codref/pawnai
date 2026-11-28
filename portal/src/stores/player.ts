// One audible session at a time. Claiming playback pauses every other player.
import { defineStore } from "pinia";
import { ref } from "vue";

export const usePlayerStore = defineStore("player", () => {
  const owner = ref<string | null>(null);

  function claim(id: string): void {
    owner.value = id;
  }

  function release(id: string): void {
    if (owner.value === id) owner.value = null;
  }

  return { owner, claim, release };
});
