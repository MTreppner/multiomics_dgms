library(dplyr)
library(ggplot2)
library(umap)
library(tibble)
library(gridExtra)
library(grid)

#setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
setwd("/Users/admin/Desktop/PhD/multi_omics_review/latent_and_meta")
csv.files = list.files(path = "/Users/admin/Desktop/PhD/multi_omics_review/latent_and_meta", pattern = "csv$", full.names = TRUE)

firstup <- function(x) {
  substr(x, 1, 1) <- toupper(substr(x, 1, 1))
  x
}

for (csv.file in csv.files){
  df <- read.csv(csv.file, row.names=1)
  
  df.meta <- df[, 11:ncol(df)]
  df.meta$ID <- row.names(df.meta)
  
  set.seed(142)
  umap_fit <- df[, 1:10] %>%
    scale() %>% 
    umap()
  
  umap_df <- umap_fit$layout %>%
    as.data.frame()%>%
    rename(UMAP1="V1",
           UMAP2="V2") %>%
    tibble::rownames_to_column(var = "ID") %>%
    inner_join(df.meta, by="ID")
  
  numeric.cols <- colnames(umap_df)[grepl('^pseudotime', colnames(umap_df))]
  umap_df[numeric.cols] <- sapply(umap_df[numeric.cols], as.numeric)
  
  factor.cols <- c("cell_type", "batch", "site", "phase", "modality", "leiden")
  umap_df[factor.cols] <- lapply(umap_df[factor.cols], factor)  ## as.factor() could also be used
  
  
  ggplots <- list()
  meta.cols <- setdiff(colnames(umap_df), c("UMAP1", "UMAP2", "ID", "modality"))

  settingString <- sub(".csv", "", basename(csv.file))
  dataType <- gsub(".*dataType_(.+)_tool.*", "\\1", settingString)
  tool <- gsub(".*tool_(.+)_cellNumber.*", "\\1", settingString)
  cellNumber <- gsub(".*cellNumber_(.+)_replicate.*", "\\1", settingString)
  replicate <- gsub(".*replicate_(.+)$", "\\1", settingString)
  
  if (cellNumber == "500"){
    pointsize <- 0.8
  } else if (cellNumber == "1000"){
    pointsize <- 0.7
  } else if (cellNumber == "2500"){
    pointsize <- 0.6
  } else if (cellNumber == "5000"){   
    pointsize <- 0.5
  } else if (cellNumber == "10000"){  
    pointsize <- 0.4
  }
  
  #umap_df <- umap_df[, c("ID", "UMAP1", "UMAP2", "cell_type", "batch", "site", "pseudotime_order_GEX", 
  #                       "pseudotime_order_ADT", "phase", "modality", "leiden")]
  meta.cols <- setdiff(colnames(umap_df), c("UMAP1", "UMAP2", "ID", "modality"))
  
  ggplotsAndLegends <- lapply(meta.cols, function(colname, umap_df, cellNumber){
    gg <- ggplot2::ggplot(umap_df, 
                          aes(x = UMAP1, y = UMAP2, color = get(colname)))+
      ggplot2::geom_point(alpha = 0.4, size=pointsize, shape=16)+
      ggplot2::labs(x = "UMAP1",
                    y = "UMAP2",
                    title = firstup(gsub("_", " ", colname))) +
      ggplot2::theme_bw() +
      ggplot2::theme(legend.title=element_blank(), plot.title = element_text(face="bold")) 
    
    if (!is.numeric(umap_df[[colname]])){
      gg <- gg + guides(color = guide_legend(override.aes = list(size=3)))
    }

    #ggsave("UMAP_plot_example1.pdf", gg, width = 14, height=6)
    legend <- cowplot::get_legend(gg)
    gg <- gg + theme(legend.position = "none")
    list(gg, legend)
    
  }, umap_df=umap_df, cellNumber=cellNumber)
  
  ggplotsAndLegends <- unlist(ggplotsAndLegends, recursive = FALSE)
  ggplots.arranged <- gridExtra::grid.arrange(grobs = ggplotsAndLegends,  ncol = 2,  widths = c(2,2) 
                                              #top = grid::textGrob(paste0(dataType, ", ", tool, ", Number of cells: ", cellNumber, ", Replicate: ", replicate), 
                                              #                     gp=grid::gpar(fontsize=15,font=2))
                                              )
  ggsave(paste0("umap_", sub(".csv", "", basename(csv.file)), ".pdf"), ggplots.arranged,
         width =15, height = 35)
}

session <- sessionInfo()
sink("umap_sessionInfo.txt")
print(session)
sink()