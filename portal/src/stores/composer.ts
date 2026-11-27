import { defineStore } from "pinia";
import { ref } from "vue";

// Per-chat composer state other parts of the UI can feed: context notes
// (vault files attached to the next turn) and quotes to insert in the draft.
export const useComposerStore = defineStore("composer", () => {
  const notes = ref<Record<string, string[]>>({});
  /** Quotes waiting for the chat's composer to pick them up. */
  const quotes = ref<Record<string, string[]>>({});

  function notesFor(chatId: string): string[] {
    return notes.value[chatId] ?? [];
  }

  function addNote(chatId: string, path: string): void {
    const cur = notesFor(chatId);
    if (!cur.includes(path)) notes.value = { ...notes.value, [chatId]: [...cur, path] };
  }

  function removeNote(chatId: string, path: string): void {
    notes.value = { ...notes.value, [chatId]: notesFor(chatId).filter((p) => p !== path) };
  }

  function clearNotes(chatId: string): void {
    const { [chatId]: _drop, ...rest } = notes.value;
    notes.value = rest;
  }

  function addQuote(chatId: string, text: string): void {
    quotes.value = { ...quotes.value, [chatId]: [...(quotes.value[chatId] ?? []), text] };
  }

  /** Take (and clear) the pending quotes for a chat. */
  function takeQuotes(chatId: string): string[] {
    const q = quotes.value[chatId] ?? [];
    if (q.length) {
      const { [chatId]: _drop, ...rest } = quotes.value;
      quotes.value = rest;
    }
    return q;
  }

  return { notes, quotes, notesFor, addNote, removeNote, clearNotes, addQuote, takeQuotes };
});
