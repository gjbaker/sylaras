suppressMessages({
  library("sleuth")
})

library("plyr")
library("dplyr")
library("tidyverse")

sample_id <- dir(file.path("/Users/gjbaker/projects/gbm_immunosuppression/RNAseq/data/sleuth", "abundance_h5"))

kal_dirs <- file.path("/Users/gjbaker/projects/gbm_immunosuppression/RNAseq/data/sleuth", "abundance_h5", sample_id, "abundance.h5")

s2c <- read.table(file.path("/Users/gjbaker/projects/gbm_immunosuppression/RNAseq/data/sleuth", "metadata", "nextseq_info.txt"), header = TRUE, stringsAsFactors = FALSE)

s2c <- dplyr::select(s2c, sample = sample, condition, type, replicate)

s2c <- dplyr::mutate(s2c, path = kal_dirs)

so <- sleuth_prep(s2c, extra_bootstrap_summary = TRUE)

mart <- biomaRt::useMart(biomart = "ENSEMBL_MART_ENSEMBL", dataset = "mmusculus_gene_ensembl", host = 'ensembl.org')

t2g <- biomaRt::getBM(attributes = c("ensembl_transcript_id", "ensembl_gene_id", "external_gene_name"), mart = mart)

t2g <- dplyr::rename(t2g, target_id = ensembl_transcript_id, ens_gene = ensembl_gene_id, ext_gene = external_gene_name)

so <- sleuth_prep(s2c, target_mapping = t2g)

so <- sleuth_fit(so, ~condition, 'full')

so <- sleuth_fit(so, ~1, 'reduced')

so <- sleuth_lrt(so, 'reduced', 'full')

models(so)

sleuth_table_lrt <- sleuth_results(so, 'reduced:full', 'lrt', show_all = FALSE)

sleuth_significant_lrt <- dplyr::filter(sleuth_table_lrt, qval <= 1.0)

head(sleuth_significant_lrt[order(sleuth_significant_lrt$pval, decreasing=FALSE), ], 100)

dev.new()
plot_pca(so, pc_x=1L, pc_y = 2L, use_filtered = TRUE, units = "est_counts", text_labels=TRUE, color_by = "condition", point_size = 7, point_alpha = 0.8) + scale_x_continuous(limits = c(-40000, 60000)) + scale_y_continuous(limits = c(-40000, 60000))
dev.copy(pdf, '/Users/gjbaker/projects/gbm_immunosuppression/RNAseq/data/sleuth/pca_min2and7.pdf')
dev.off()

dev.new()
plot_pc_variance(so, use_filtered = TRUE, units = 'est_counts', pca_number = NULL, scale = FALSE, PC_relative = NULL)
dev.copy(pdf, '/Users/gjbaker/projects/gbm_immunosuppression/RNAseq/data/sleuth/scree_min2and7.pdf')
dev.off()
pg <- ggplot_build(plot_pc_variance(so, use_filtered = TRUE, units = 'est_counts', pca_number = NULL, scale = FALSE, PC_relative = NULL))


dev.new()
plot_loadings(so, use_filtered = FALSE, sample = NULL, pc_input = 1, units = "est_counts", pc_count = NULL, scale = FALSE, pca_loading_abs = TRUE)
dev.copy(pdf, '/Users/gjbaker/projects/gbm_immunosuppression/RNAseq/data/sleuth/PC1_loadings_min2and7.pdf')
dev.off()

so <- sleuth_wt(so, which_beta = 'conditionexperimental', which_model = 'full')

sleuth_table_wt <- sleuth_results(so, test = 'conditionexperimental', test_type = 'wt', which_model = "full")

sleuth_significant_wt <- dplyr::filter(sleuth_table_wt, qval <= 0.05)

sleuth_significant_wt <- head(sleuth_significant_wt[order(sleuth_significant_wt$qval, decreasing=FALSE), ], 100)

write.table(sleuth_significant_wt, file = "/Users/gjbaker/projects/gbm_immunosuppression/RNAseq/data/sleuth/sleuth_sig_min2and7.csv", row.names = FALSE, na = "", col.names = TRUE, sep = ",")


sleuth_allgenes <- sleuth_table_wt[order(sleuth_table_wt$b, decreasing=FALSE), ]

sleuth_allgenes <- sleuth_allgenes[c("ext_gene", "b")][complete.cases(sleuth_allgenes), ]

sleuth_allgenes <- ddply(sleuth_allgenes, ~ext_gene, summarise, mean=mean(b))

sleuth_allgenes <- sleuth_allgenes[order(sleuth_allgenes$mean, decreasing=FALSE), ]

write.table(sleuth_allgenes, file = "/Users/gjbaker/projects/gbm_immunosuppression/RNAseq/gsea/rnk_files/sleuth_allgenes_min2and7.rnk", row.names = FALSE, na = "", col.names = FALSE, sep = "\t", quote = FALSE)


normalized_TPM = sleuth_to_matrix(so, "obs_norm", "tpm")

normalized_TPM = sleuth_to_matrix(so, "obs_norm", "tpm")
write.table(normalized_TPM, file = "/Users/gjbaker/projects/gbm_immunosuppression/RNAseq/data/sleuth/normalized_TPM_matrix.tsv", row.names = TRUE, na = "", col.names = TRUE, sep = "\t", quote = FALSE)
