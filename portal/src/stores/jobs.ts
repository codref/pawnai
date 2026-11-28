// Live job status for prep and capture-enrich spinners.
// `watchJob` resolves when the job leaves the running states.
import { defineStore } from "pinia";
import { ref } from "vue";
import { api, type Job } from "@/api";
import { isAbort } from "@/lib/errors";

const TERMINAL = new Set(["done", "blocked", "deleted", "review"]);

function isTerminal(status: string): boolean {
  return TERMINAL.has(status);
}

export const useJobsStore = defineStore("jobs", () => {
  const jobs = ref<Record<string, Job>>({});
  const waiters = new Map<string, Array<(job: Job) => void>>();
  let ctl: AbortController | null = null;
  let pumping = false;

  function deliver(job: Job): void {
    jobs.value = { ...jobs.value, [job.id]: job };
    if (!isTerminal(job.status)) return;
    const pending = waiters.get(job.id) ?? [];
    waiters.delete(job.id);
    for (const resolve of pending) resolve(job);
  }

  async function pump(signal: AbortSignal): Promise<void> {
    while (!signal.aborted) {
      try {
        await api().jobEvents(deliver, signal);
      } catch (e) {
        if (isAbort(e) || signal.aborted) return;
        await new Promise((r) => setTimeout(r, 2000));
      }
    }
  }

  function ensure(): void {
    if (pumping) return;
    pumping = true;
    ctl = new AbortController();
    const signal = ctl.signal;
    void pump(signal).finally(() => {
      if (ctl?.signal === signal) {
        pumping = false;
        ctl = null;
      }
    });
  }

  /** Resolves with the job once it is done, blocked, deleted, or in review. */
  function watchJob(id: string): Promise<Job> {
    const current = jobs.value[id];
    if (current && isTerminal(current.status)) return Promise.resolve(current);
    ensure();
    return new Promise((resolve) => {
      const list = waiters.get(id) ?? [];
      list.push(resolve);
      waiters.set(id, list);
      void api()
        .getJob(id)
        .then(deliver)
        .catch(() => {
          /* not created yet; the event stream will deliver it */
        });
    });
  }

  function stop(): void {
    ctl?.abort();
    ctl = null;
    pumping = false;
  }

  return { jobs, watchJob, stop };
});
