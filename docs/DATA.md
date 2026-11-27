# Included input data

## PDAC A-1

- Study: Moncada et al., *Integrating microarray-based spatial transcriptomics
  and single-cell RNA-seq reveals tissue architecture in pancreatic ductal adenocarcinomas*.
- GEO series [GSE111672](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE111672);
  sample [GSM3036911, PDAC-A ST1](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM3036911).
- Technology: original spatial transcriptomics array, **not 10x Visium**.
- `counts.tsv.gz`: count table from the project's `GSM3036911.tsv`, gzip-compressed
  for transport. Spots are rows and 25,753 gene columns are retained before filtering.
  Compression does not normalize, impute or otherwise alter count values.
- `spot_locations.tsv`: the original `locs-raw-sorted.txt`, containing 428 registered
  tissue spots and manual reference-domain annotations. Matrix rows are selected
  and ordered by the spot IDs in this file; extra array spots are not used.
- `he_image.png`: the project's registered 640 × 632 working H&E image, copied
  from `data/PDAC/data/ST-imgs/he-raw.png`. It is not the original full-size GEO image.

**PDAC coordinate convention:** the source columns named `x,y` mean image row
and image column. The reader explicitly renames them to `pixel_y,pixel_x` and
stores `.obsm['spatial'] = [pixel_x, pixel_y]`, matching the original PDAC notebook.
The supplied working image uses scale 1.0. Manual domain labels are retained in
input metadata; they do not enter gene selection, graph construction or training.

## Breast Cancer Visium

- Sample: **V1_Breast_Cancer_Block_A_Section_1**.
- Provider: 10x Genomics, *Breast Cancer: Ductal Carcinoma In Situ, Lobular Carcinoma
  In Situ, Invasive Carcinoma (Block A Section 1)*.
- [Dataset page](https://support.10xgenomics.com/spatial-gene-expression/datasets/1.1.0/V1_Breast_Cancer_Block_A_Section_1)
  and [provider's sample summary](https://cf.10xgenomics.com/samples/spatial-exp/1.0.0/V1_Breast_Cancer_Block_A_Section_1/V1_Breast_Cancer_Block_A_Section_1_web_summary.html).
- `filtered_feature_bc_matrix.h5`: the project's genuine 10x count HDF5, not a
  normalized H5AD or a model-generated matrix. The reader preserves gene IDs
  and symbols and uses Gene Expression features if feature types are present.
- `spatial/tissue_positions_list.csv`: headerless Space Ranger table with columns
  barcode, in_tissue, array_row, array_col, pixel_row, pixel_col. Counts and
  positions are aligned by barcode, and `in_tissue == 1` retains 3,798 tissue spots.
- `spatial/tissue_hires_image.png`: 2,000 × 2,000 image used for the six RGB features.
- `spatial/scalefactors_json.json`: official registration/scaling metadata.
- `spatial/tissue_lowres_image.png`: provider's display image, included as an
  input asset; the default feature extractor uses the hires image.

**Visium coordinate convention:** `.obsm['spatial'] = [pixel_col, pixel_row]` in
full-resolution registered image units. Multiply by `tissue_hires_scalef =
0.08250825` only when accessing the hires image. Do not multiply twice.
A 15 × 15 patch is measured in pixels of the supplied hires image, not fullres pixels.
PDAC and Visium therefore have different physical patch footprints.

## Input integrity and reuse

`data/manifest.json` records source project paths, packaged file sizes and SHA-256
fingerprints. These identify the supplied inputs; they are not computed model
results. Source paths are provenance only and are never accessed by the runtime.
The files needed by either default workflow are already bundled, so running the
code does not require a download or access to the authors' server.

Keep counts, coordinates and images from the same sample together. Replacing an
image with a higher-resolution slide requires an updated coordinate transform;
merely replacing its filename would produce incorrectly registered patches.
Original provider terms and attribution continue to apply to the data and images.
