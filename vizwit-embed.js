
<div style="min-height: 230px">
  <script class="vizwit" type="application/json">
{
  "title": "Census Tract",
  "provider": "carto",
  "domain": "phl.carto.com",
  "dataset": "li_unsafe",
  "chartType": "choropleth",
  "groupBy": "censustract",
  "boundaries": "https://phl.carto.com/api/v2/sql?q=select+*+from+census_tracts_2010&format=geojson",
  "boundariesLabel": "namelsad10",
  "boundariesId": "name10_padded",
  "triggerField": "censustract",
  "baseFilters": [],
  "filters": {},
  "aggregateFunction": "count"
}
  </script>
</div>
