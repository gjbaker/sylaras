#!/bin/bash

cd /Users/gjbaker/projects/gbm_immunosuppression/RNAseq/data/gsea_output

# Broad Institute GSEA_Preranked usage:
java -Xmx2048m -cp /Users/gjbaker/projects/gbm_immunosuppression/RNAseq/gsea/gsea3.0.jar xtools.gsea.GseaPreranked -rnk /Users/gjbaker/projects/gbm_immunosuppression/RNAseq/gsea/rnk_files/edgeR_allgenes_min2and7.rnk -gmx /Users/gjbaker/projects/gbm_immunosuppression/RNAseq/gsea/gmx_files/MousePath.GO.gmt.gmt -norm meandiv -nperm 1000 -scoring_scheme weighted -rpt_label edgeR_allgenes_min2and7 -create_svgs false -make_sets true -plot_top_x 3 -rnd_seed timestamp -set_max 500 -set_min 15 -zip_report true -out /Users/gjbaker/projects/gbm_immunosuppression/RNAseq/data/gsea_output
