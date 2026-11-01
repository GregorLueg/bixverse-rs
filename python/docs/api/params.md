# Parameters

One frozen dataclass per parameter set, flat, the way Rust sees it. Grouped by the
module that takes them; `Shared` goes to more than one.

## Shared

::: bixverse.params
    options:
      show_root_toc_entry: false
      show_root_heading: false
      heading_level: 3
      members:
        - BbknnParams
        - FastmnnParams
        - HarmonyParams
        - HarmonyV2Params
        - HvgParams
        - KmeansDefaults
        - KnnParams
        - NeighboursParams
        - PcaParams
        - WnnParams

## Input and output

::: bixverse.params
    options:
      show_root_toc_entry: false
      show_root_heading: false
      heading_level: 3
      members:
        - MtxIoParams
        - SyntheticCellsweepParams
        - SyntheticDataAdtParams
        - SyntheticDataParams
        - SyntheticDialogueParams

## Preprocessing

::: bixverse.params
    options:
      show_root_toc_entry: false
      show_root_heading: false
      heading_level: 3
      members:
        - AprParams
        - HvgDefaults
        - HvgScranDefaults
        - MagicParams
        - PcaDefaults
        - SctransformParams

## Neighbours and clusters

::: bixverse.params
    options:
      show_root_toc_entry: false
      show_root_heading: false
      heading_level: 3
      members:
        - FastClusterDefault
        - FastClusterParams

## Batch correction

::: bixverse.params
    options:
      show_root_toc_entry: false
      show_root_heading: false
      heading_level: 3
      members:
        - SeuratCcaParams
        - SeuratRpcaParams

## Doublets and ambient RNA

::: bixverse.params
    options:
      show_root_toc_entry: false
      show_root_heading: false
      heading_level: 3
      members:
        - BoostParams
        - CellsweepParams
        - EmptyDropletsParams
        - MinQualityParams
        - NormDoubletsDefaults
        - ScdblfinderParams
        - ScrubletParams

## Differential expression and abundance

::: bixverse.params
    options:
      show_root_toc_entry: false
      show_root_heading: false
      heading_level: 3
      members:
        - EdgerQlParams
        - MeldParams
        - NebulaParams

## Annotation

::: bixverse.params
    options:
      show_root_toc_entry: false
      show_root_heading: false
      heading_level: 3
      members:
        - SctypeCellsParams
        - SymphonyMapParams

## Meta cells

::: bixverse.params
    options:
      show_root_toc_entry: false
      show_root_heading: false
      heading_level: 3
      members:
        - BtMetacellsParams
        - KnnDefaults
        - Metacells2DacParams
        - Metacells2Params
        - SeacellsParams
        - SupercellParams

## Trajectories

::: bixverse.params
    options:
      show_root_toc_entry: false
      show_root_heading: false
      heading_level: 3
      members:
        - BranchSelectionParams
        - GeneTrendsParams
        - PalantirParams

## Cell trees

::: bixverse.params
    options:
      show_root_toc_entry: false
      show_root_heading: false
      heading_level: 3
      members:
        - BonsaiParams

## Gene sets and regulons

::: bixverse.params
    options:
      show_root_toc_entry: false
      show_root_heading: false
      heading_level: 3
      members:
        - AucellParams
        - CistargetParams
        - HotspotParams
        - LmActivityParams
        - ScenicBinariseParams
        - ScenicExtraTreesDefaults
        - ScenicGradientBoostingDefaults
        - ScenicParams
        - ScenicRandomForestDefaults
        - VisionParams

## NMF

::: bixverse.params
    options:
      show_root_toc_entry: false
      show_root_heading: false
      heading_level: 3
      members:
        - NmfConsensusParams
        - NmfHalsParams

## Multi-modal

::: bixverse.params
    options:
      show_root_toc_entry: false
      show_root_heading: false
      heading_level: 3
      members:
        - DsbParams
        - KnnGraphDefaults

## Ligand-receptor

::: bixverse.params
    options:
      show_root_toc_entry: false
      show_root_heading: false
      heading_level: 3
      members:
        - CellphonedbParams
        - LigandTargetParams

## DIALOGUE

::: bixverse.params
    options:
      show_root_toc_entry: false
      show_root_heading: false
      heading_level: 3
      members:
        - DialogueHlmParams
        - DialoguePmdParams
        - DialogueRefineParams
