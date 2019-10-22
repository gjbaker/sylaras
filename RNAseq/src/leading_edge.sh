#!/bin/bash

cd /Users/gjbaker/projects/gbm_immunosuppression/RNAseq/data/gsea_output

# Broad Institute Leading Edge Analysis usage:
java -Xmx2048m -cp /Users/gjbaker/projects/gbm_immunosuppression/RNAseq/gsea/gsea3.0.jar xtools.gsea.LeadingEdgeTool -dir /Users/gjbaker/projects/gbm_immunosuppression/RNAseq/data/gsea_output/edgeR_allgenes_min2and7.GseaPreranked.1545342129922 -gsets GO_BP_MM_ANTIGEN_PROCESSING_AND_PRESENTATION, GO_BP_MM_IMMUNE_RESPONSE, GO_CC_MM_MEMBRANE, GO_CC_MM_MHC_CLASS_I_PROTEIN_COMPLEX -rpt_label edgeR_allgenes_min2and7 -imgFormat png -extraPlots true -out /Users/gjbaker/projects/gbm_immunosuppression/RNAseq/data/gsea_output
