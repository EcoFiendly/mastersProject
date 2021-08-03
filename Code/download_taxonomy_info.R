library('rredlist')

rl_citation()

out <- rl_sp(all = TRUE)
length(out)
vapply(out, "[[", 1, "count")
all_df <- do.call(rbind, lapply(out, "[[", "result"))
head(all_df)
NROW(all_df)
write.csv(all_df, '../Data/taxonomy_info.csv')
