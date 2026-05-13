# Dataset Statement

This repository includes the sampled InSAR domain files used by the formal
InSAR-DA experiments. The files are compact, preprocessed NumPy archives under
`data/domains_10k_50x50`.

## Included Domains

| Domain | Hazard type | Source tag | Included file |
| --- | --- | --- | --- |
| `Subsidence_Po_River_Piacenza` | subsidence | `TS_INGV_ITA_S-1_2015_2022_UAZ` | `data/domains_10k_50x50/Subsidence_Po_River_Piacenza/sampled_10000_grid50x50_seed42.npz` |
| `Subsidence_Como` | subsidence | `TS_INGV_ITA_CSK_2010_2019_FCQ` | `data/domains_10k_50x50/Subsidence_Como/sampled_10000_grid50x50_seed42.npz` |
| `Volcano_Aeolian_Islands` | volcano | `TS_INGV_ITA_S-1_2016_2023_DDZ` | `data/domains_10k_50x50/Volcano_Aeolian_Islands/sampled_10000_grid50x50_seed42.npz` |
| `Volcano_Campi_Flegrei` | volcano | `TS_INGV_ITA_S-1_2016_2023_DCL` | `data/domains_10k_50x50/Volcano_Campi_Flegrei/sampled_10000_grid50x50_seed42.npz` |
| `Landslide_Saline_Joniche` | landslide | `TS_INGV_ITA_CSK_2015_2023_FIW` | `data/domains_10k_50x50/Landslide_Saline_Joniche/sampled_10000_grid50x50_seed42.npz` |
| `Landslide_Bagnoregio` | landslide | `TS_INGV_ITA_CSK_2012_2020_OJG` | `data/domains_10k_50x50/Landslide_Bagnoregio/sampled_10000_grid50x50_seed42.npz` |

## File Format

Each `.npz` file contains:

- `displacement_full`: `float32` array with shape `(num_points, num_observations)`.
- `dates`: `int32` observation-step index array.
- `latlon`: `float32` array with shape `(num_points, 2)`.
- `optional__point_id`: sampled point identifiers retained from the processed
  source product when available.
- `optional__*`: auxiliary fields retained from the processed source product.
- `metadata_json`: JSON metadata for the sampled public archive.

The sampled files contain 10,000 points per domain selected with a 50 x 50 grid
sampling layout and seed `42`.

## Benchmark Registry And Splits

The formal benchmark reads the sampled-domain registry from:

```text
data/datasets_public_true_types_obs_step_final_10k_50x50.yaml
```

The registry and configuration define 24 transfer tasks across LODO, IHT, and
CHT protocols. The main benchmark configuration is:

```text
configs/main.yaml
```

The task protocol uses 20 input steps, five forecast steps, and stride-five
window subsampling for the formal training and test partitions. Source windows
are split 7:3 into source-training and source-validation time bands. Target
windows are split into adaptation, validation, and test time bands at 5:2:3.
Target adaptation-label rates are point-level budgets of 0.5%, 1%, 2.5%, and
5% of the 10,000 target-domain points. Target-label sampling uses the run seeds
`42`, `43`, and `44`, separately from the sampled-domain grid seed `42`.

The target validation band is used for protocol-level model selection in
target-aware methods and is not part of the target adaptation-label budget.
The target test band is not used for training, validation, normalization, or
model selection.

## Provenance And License

The source products come from the INGV InSAR ground displacement time-series
archive:

```text
InSAR Working Group. (2013). InSAR ground displacement time series.
Istituto Nazionale di Geofisica e Vulcanologia (INGV).
https://doi.org/10.13127/insar/ts
```

The DOI landing page states that the archive can be accessed free of charge and
that contents are distributed under the Creative Commons Attribution license
linked there as CC BY 4.0. The sampled `.npz` files in this repository are
derived from those public source products and must retain the INGV/InSAR Working
Group attribution.

| Source tag | Official DOI or URL | Source-data license | Required attribution |
| --- | --- | --- | --- |
| `TS_INGV_ITA_S-1_2015_2022_UAZ` | https://doi.org/10.13127/insar/ts | Creative Commons Attribution 4.0 International (CC BY 4.0) | InSAR Working Group. (2013). InSAR ground displacement time series. Istituto Nazionale di Geofisica e Vulcanologia (INGV). https://doi.org/10.13127/insar/ts |
| `TS_INGV_ITA_CSK_2010_2019_FCQ` | https://doi.org/10.13127/insar/ts | Creative Commons Attribution 4.0 International (CC BY 4.0) | InSAR Working Group. (2013). InSAR ground displacement time series. Istituto Nazionale di Geofisica e Vulcanologia (INGV). https://doi.org/10.13127/insar/ts |
| `TS_INGV_ITA_S-1_2016_2023_DDZ` | https://doi.org/10.13127/insar/ts | Creative Commons Attribution 4.0 International (CC BY 4.0) | InSAR Working Group. (2013). InSAR ground displacement time series. Istituto Nazionale di Geofisica e Vulcanologia (INGV). https://doi.org/10.13127/insar/ts |
| `TS_INGV_ITA_S-1_2016_2023_DCL` | https://doi.org/10.13127/insar/ts | Creative Commons Attribution 4.0 International (CC BY 4.0) | InSAR Working Group. (2013). InSAR ground displacement time series. Istituto Nazionale di Geofisica e Vulcanologia (INGV). https://doi.org/10.13127/insar/ts |
| `TS_INGV_ITA_CSK_2015_2023_FIW` | https://doi.org/10.13127/insar/ts | Creative Commons Attribution 4.0 International (CC BY 4.0) | InSAR Working Group. (2013). InSAR ground displacement time series. Istituto Nazionale di Geofisica e Vulcanologia (INGV). https://doi.org/10.13127/insar/ts |
| `TS_INGV_ITA_CSK_2012_2020_OJG` | https://doi.org/10.13127/insar/ts | Creative Commons Attribution 4.0 International (CC BY 4.0) | InSAR Working Group. (2013). InSAR ground displacement time series. Istituto Nazionale di Geofisica e Vulcanologia (INGV). https://doi.org/10.13127/insar/ts |

The code in this repository is licensed separately under `LICENSE`. The dataset
archives are subject to the source-data attribution requirements above. Do not
remove provider attribution when redistributing derived data.

## Regenerating Or Replacing Data

The formal experiments read the dataset registry from:

```text
data/datasets_public_true_types_obs_step_final_10k_50x50.yaml
```

To use a different public data release, place compatible `.npz` files under
`data/` and update the registry paths and source tags. Keep raw private data,
temporary exports, and unreleased provider files outside the repository or under
ignored directories such as `data/raw/` or `data/private/`.
