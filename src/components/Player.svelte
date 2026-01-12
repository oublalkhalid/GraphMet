<script>
  import Pause from 'carbon-icons-svelte/lib/PauseFilled.svelte';
  import Play from 'carbon-icons-svelte/lib/PlayFilled.svelte';
  import { ParticleDataset } from '../datasets.js';
  import { tick } from 'svelte';

  export let validDates;
  export let griddedDataset;
  export let particleDataset;
  export let date;
  export let sliding = false;
  export let fps = 5;

  let playing = false;
  let loading = false;
  let controller;
  let originalParticleDataset = particleDataset;
  let timeoutID;
  let value = 0;
  let repeat = true; 
  let syncingValueWithDate = false;

  $: date, handleDateUpdate();
  $: maxValue = validDates.length - 1;
  $: time = 1000 / (fps || 1);
  $: value, slideDate();

  async function play() {
    if (playing) return;
    originalParticleDataset = particleDataset;
    particleDataset = ParticleDataset.none;
    
    await tick();
    playing = true;
    loading = true;

    controller = new AbortController();
    let { signal } = controller;

    try {
      // Fetch only the next batch to prevent browser hang
      const nextBatch = validDates.slice(value, value + 10);
      await Promise.all(nextBatch.map(d => griddedDataset.fetchData(d, signal)));
    } catch (e) {
      if (e.name !== 'AbortError') console.error(e);
      return;
    } finally {
      loading = false;
    }

    if (value >= maxValue || value < 0) value = 0;
    timeoutID = window.setTimeout(loopDate, time);
  }

  function loopDate() {
    if (!playing) return;
    value = (value + 1) > maxValue ? 0 : value + 1;
    if (value === 0 && !repeat) {
      pause();
      return;
    }
    timeoutID = window.setTimeout(loopDate, time);
  }

  function pause() {
    if (!playing) return;
    if (loading && controller) controller.abort();
    window.clearTimeout(timeoutID);
    playing = false;
    loading = false;
    particleDataset = originalParticleDataset;
  }

  async function slideDate() {
    if (syncingValueWithDate || !validDates[value]) return;
    sliding = true;
    date = validDates[value];
    await tick();
    sliding = false;
  }

  async function handleDateUpdate() {
    if (sliding || !date) return;
    syncingValueWithDate = true;
    const newIndex = validDates.findIndex(d => d.getTime() === date.getTime());
    if (newIndex !== -1) value = newIndex;
    await tick();
    syncingValueWithDate = false;
  }
</script>

<button
  class="v-play-btn"
  class:is-loading={loading}
  on:click={playing ? pause : play}
  aria-label="Play/Pause"
>
  {#if playing}
    <Pause size={24} />
  {:else}
    <Play size={24} />
  {/if}
</button>

<style>
  .v-play-btn {
    all: unset;
    display: flex;
    align-items: center;
    justify-content: center;
    width: 34px;
    height: 34px;
    background: white !important;
    color: #3366ff !important;
    border-radius: 50%;
    cursor: pointer;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
    transition: transform 0.1s ease;
  }

  .v-play-btn:active {
    transform: scale(0.95);
  }

  .is-loading {
    animation: v-pulse 1.5s infinite ease-in-out;
  }

  @keyframes v-pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.4; }
  }

  .v-play-btn :global(svg) {
    width: 20px;
    height: 20px;
    fill: currentColor;
  }
</style>