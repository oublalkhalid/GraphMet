<script>
  import { fly, fade } from 'svelte/transition';
  // Removed PinIcon import
  import { labelByName, prettyValue } from '../units.js';
  import { prettyLatLon } from '../utility.js';

  export let pins;
  export let pin;
  export let forwardProjectionFunction;
  export let griddedName;
  export let griddedData;
  export let griddedUnit;

  let lonLat = [pin.longitude, pin.latitude];
  let hovering = false;

  let x = 0, y = 0;
  $: point = forwardProjectionFunction(lonLat)
  $: clip = point === null;
  $: if (point) [x, y] = point;
  $: value = griddedData.get(lonLat);
  $: label = labelByName(value, griddedName);
</script>


<div
  class="pin"
  class:clip
  style="left: {x}px; top: {y}px;"
  in:fly="{{ y: -150, duration: 250 }}"
  out:fade="{{ duration: 250 }}"
>
  <button
    class="marker"
    style="z-index: {-1e8 + Math.round(y * 8)}"
    on:click={() => pins = pins.filter(p => p !== pin)}
    on:focus={() => hovering = true}
    on:blur={() => hovering = false}
    on:mouseenter={() => hovering = true}
    on:mouseleave={() => hovering = false}
  >
    <div class="turbine">
      <div class="blades"></div>
      <div class="tower"></div>
    </div>
  </button>

  {#if hovering}
    <div class="caption">
      <span class="plain">
        {#if pin.label}
          {pin.label}
        {:else}
          Location {pin.id}
        {/if}
        <br>
      </span>
      <strong class="bold">
        {prettyValue(value, griddedData.originalUnit, griddedUnit, label)}
      </strong><br>
      <small class="plain">
        {prettyLatLon(pin.latitude, pin.longitude)}
      </small>
    </div>
  {/if}
</div>


<style>
  div.pin {
    position: absolute;
    pointer-events: auto;
  }

  div.pin.clip {
    display: none;
  }

  div.pin > :global(*) {
    position: absolute;
  }

  /* Turbine Container Positioning */
  div.pin > :global(.marker) {
    top: -35px; /* Aligns the bottom of the tower to the coordinate */
    left: -15px;
    cursor: pointer;
    display: block;
    padding: 0;
    border: none;
    background-color: transparent;
    filter: drop-shadow(0 0 4px rgba(0,0,0,0.5));
  }

  /* The Turbine Tower (Pole) */
  .tower {
    position: absolute;
    bottom: 0;
    left: 50%;
    width: 2px;
    height: 22px;
    background: white;
    transform: translateX(-50%);
  }

  /* The Spinning Blades */
  .blades {
    position: absolute;
    top: 0;
    left: 50%;
    width: 30px;
    height: 30px;
    margin-left: -15px;
    border-radius: 50%;
    /* Draws three blades using a gradient */
    background: conic-gradient(
      from 0deg,
      transparent 0deg 5deg,
      white 5deg 15deg,
      transparent 15deg 120deg,
      white 120deg 130deg,
      transparent 130deg 240deg,
      white 240deg 250deg,
      transparent 250deg
    );
    animation: turbine-spin 2s linear infinite;
  }

  @keyframes turbine-spin {
    from { transform: rotate(0deg); }
    to { transform: rotate(360deg); }
  }

  .caption {
    width: max-content;
    background-color: beige;
    border-radius: .5rem;
    box-shadow: 0 0 10px #000;
    font-size: .8rem;
    line-height: 1.2;
    padding: .25rem .5rem;
    top: 0.25rem;
    left: 1.25rem; /* Moved right to avoid overlapping the turbine */
    z-index: 10;
  }

  .plain {
    color: #6c757d;
  }

  .bold {
    color: #000;
    font-weight: bolder;
  }
</style>