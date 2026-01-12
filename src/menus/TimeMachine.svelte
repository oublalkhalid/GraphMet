<script>
  // Use ../ to go up one folder from 'menus' to 'src', then into 'components'
  import TimeJumper from '../components/TimeJumper.svelte';
  import Player from '../components/Player.svelte';
  
  // If Calendar is inside src/components/calendar/
  import Calendar from '../components/calendar/Calendar.svelte';


  export let date;
  export let utc;
  export let timeDataset;
  export let particleDataset;

  let showCalendar = false;

  function move(steps) {
    const dates = timeDataset.validDates;
    const currentIndex = dates.findIndex(d => d.getTime() === date.getTime());
    const newIndex = currentIndex + steps;
    if (newIndex >= 0 && newIndex < dates.length) {
      date = dates[newIndex];
    }
  }

  const hourTicks = ['01:00', '04:00', '07:00', '10:00', '13:00', '16:00', '19:00', '22:00'];

  $: dateString = date.toLocaleDateString('fr-FR', { 
    weekday: 'long', day: '2-digit', month: '2-digit', year: 'numeric' 
  });
</script>

<div class="v-main-container">
  <div class="v-side">
    <div class="v-block">
      <Player {utc} validDates={timeDataset.validDates} 
              griddedDataset={timeDataset} bind:date bind:particleDataset />
      <span class="v-label">Démarrer</span>
    </div>
    <div class="v-block">
      <button class="v-circle" on:click={() => move(-1)}>
        <svg viewBox="0 0 24 24" width="18" height="18"><path fill="currentColor" d="M15.41 16.59L10.83 12l4.58-4.59L14 6l-6 6 6 6 1.41-1.41z"/></svg>
      </button>
      <span class="v-label">Précédent</span>
    </div>
  </div>

  <div class="v-block">
    <button class="v-pill" on:click={() => showCalendar = !showCalendar}>
      {dateString} <span class="v-arrow">▼</span>
    </button>
    <span class="v-label">Changer la date</span>
    {#if showCalendar}
      <div class="v-popover">
        <Calendar {timeDataset} bind:date {utc} />
        <button class="v-close" on:click={() => showCalendar = false}>✕</button>
      </div>
    {/if}
  </div>

  <div class="v-timeline-stretch">
    <div class="v-slider-box">
      <TimeJumper {timeDataset} bind:date />
    </div>
    <div class="v-jalon-labels">
      {#each hourTicks as tick}
        <div class="v-tick-unit">
          <div class="v-line"></div>
          <span>{tick}</span>
        </div>
      {/each}
    </div>
  </div>

  <div class="v-side">
    <div class="v-block">
      <button class="v-circle" on:click={() => move(1)}>
        <svg viewBox="0 0 24 24" width="18" height="18"><path fill="currentColor" d="M8.59 16.59L13.17 12 8.59 7.41 10 6l6 6-6 6-1.41-1.41z"/></svg>
      </button>
      <span class="v-label">Suivant</span>
    </div>
  </div>
</div>

<style>
  .v-main-container {
    display: flex;
    align-items: flex-end;
    width: 100vw;
    height: 95px;
    padding: 0 20px 15px;
    gap: 15px;
    background: linear-gradient(transparent, rgba(0,0,0,0.85));
    position: fixed;
    bottom: 0;
    left: 0;
    z-index: 1000;
    box-sizing: border-box;
  }

  .v-side { display: flex; gap: 15px; }

  .v-block {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 5px;
  }

  .v-label {
    font-size: 10px;
    color: white;
    text-shadow: 1px 1px 2px black;
    white-space: nowrap;
  }

  .v-circle, :global(.v-block button) {
    background: white !important;
    color: #3366ff !important;
    border: none !important;
    border-radius: 50% !important;
    width: 34px !important;
    height: 34px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    cursor: pointer;
    box-shadow: 0 2px 5px rgba(0,0,0,0.3);
  }

  .v-pill {
    background: white;
    color: #3366ff;
    border: none;
    border-radius: 20px;
    padding: 6px 14px;
    font-weight: bold;
    font-size: 13px;
    cursor: pointer;
    white-space: nowrap;
  }

  .v-timeline-stretch {
    flex: 1; 
    display: flex;
    flex-direction: column;
    padding-bottom: 18px;
    min-width: 0; /* Prevents flex items from overflowing */
  }

  .v-slider-box { width: 100%; }

  .v-jalon-labels {
    display: flex;
    justify-content: space-between;
    width: 100%;
    margin-top: 8px;
  }

  .v-tick-unit { display: flex; flex-direction: column; align-items: center; }
  .v-line { width: 1px; height: 5px; background: rgba(255,255,255,0.5); }
  .v-tick-unit span { font-size: 10px; color: rgba(255,255,255,0.7); font-family: monospace; }

  .v-popover {
    position: absolute;
    bottom: 90px;
    background: white;
    border-radius: 8px;
    padding: 10px;
    box-shadow: 0 5px 15px rgba(0,0,0,0.5);
  }

  :global(.v-main-container h2), :global(.v-main-container summary), :global(.v-main-container p) {
    display: none !important;
  }
</style>