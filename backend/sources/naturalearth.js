import { download } from '../download.js';
import {
  brotli,
  hash_of_this_file,
  parent_output_dir,
  write_file_atomically,
} from '../utility.js';
import { rm } from 'fs/promises';
import mapshaper from 'mapshaper';
import { join } from 'path';

// We need two bases: physical and cultural
const physical_base = 'https://raw.githubusercontent.com/nvkelso/natural-earth-vector/v5.0.1/50m_physical/';
const cultural_base = 'https://raw.githubusercontent.com/nvkelso/natural-earth-vector/v5.0.1/50m_cultural/';

const urls = [
  physical_base + 'ne_50m_graticules_all/ne_50m_graticules_10.shp',
  physical_base + 'ne_50m_rivers_lake_centerlines.shp',
  physical_base + 'ne_50m_lakes.shp',
  physical_base + 'ne_50m_coastline.shp',
  // ADDED FOR VENTUSKY STYLE:
  cultural_base + 'ne_50m_admin_0_countries.shp', // National borders
  cultural_base + 'ne_50m_populated_places.shp',  // Cities and towns
];

export async function forage(current_state) {
  let source_hash = await hash_of_this_file(import.meta);
  if (source_hash === current_state.source_hash) {
    throw new Error('No update needed');
  }

  // Download all files (Shapefiles)
  let files = await Promise.all(urls.map(url => download(url, {}, false)));
  let output = join(parent_output_dir, 'topology.json.br');

  // MAPSHAPER LOGIC: 
  // We include 'fields=NAME' so city names are preserved in the JSON
  let cmds = `-i ${files.join(' ')} combine-files ` +
             `-filter-fields NAME,ADM0_A3 ` + 
             `-o out.json format=topojson`;

  let topojson = (await mapshaper.applyCommands(cmds))['out.json'];
  
  // Cleanup temporary downloads
  await Promise.all(files.map(file => rm(file)));

  // Compress and save
  await write_file_atomically(output, await brotli(topojson));

  return { new_state: { source_hash } };
}