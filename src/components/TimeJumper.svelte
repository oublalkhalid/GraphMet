<script>
  export let timeDataset;
  export let date;

  $: value = timeDataset.validDates.findIndex(d => d.getTime() === date.getTime());

  function handleChange(e) {
    const idx = parseInt(e.target.value);
    const newDate = timeDataset.validDates[idx];
    if (newDate) {
      date = newDate;
    }
  }
</script>

<div class="v-engine">
  <input 
    type="range" 
    min="0" 
    max={timeDataset.validDates.length - 1} 
    {value} 
    on:input={handleChange}
  />
</div>

<style>
  .v-engine { width: 100%; display: flex; align-items: center; }
  
  input[type="range"] {
    -webkit-appearance: none;
    width: 100%;
    height: 4px;
    background: rgba(255, 255, 255, 0.3);
    border-radius: 2px;
    outline: none;
    cursor: pointer;
  }

  input[type="range"]::-webkit-slider-thumb {
    -webkit-appearance: none;
    width: 14px;
    height: 14px;
    background: white;
    border-radius: 50%;
    box-shadow: 0 0 4px rgba(0,0,0,0.5);
  }
</style>