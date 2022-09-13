
library(dplyr)
library(ggplot2)
library(umap)
library(tibble)
library(gridExtra)
library(grid)
library(cowplot)

gg_color_hue <- function(n) {
  hues = seq(15, 375, length = n + 1)
  hcl(h = hues, l = 65, c = 100)[1:n]
}

getColors <- function(){
  multiome500 <- read.csv(paste0("/Users/admin/Desktop/PhD/multi_omics_review_old/latent_and_meta/dataType_Multiome_tool_Cobolt_cellNumber_500_replicate_1.csv"), row.names=1)
  multiome10000 <- read.csv(paste0("/Users/admin/Desktop/PhD/multi_omics_review_old/latent_and_meta/dataType_Multiome_tool_Cobolt_cellNumber_10000_replicate_1.csv"), row.names=1)
  CITE500 <- read.csv(paste0("/Users/admin/Desktop/PhD/multi_omics_review_old/latent_and_meta/dataType_CITE_tool_Cobolt_cellNumber_500_replicate_1.csv"), row.names=1)
  CITE10000 <- read.csv(paste0("/Users/admin/Desktop/PhD/multi_omics_review_old/latent_and_meta/dataType_CITE_tool_Cobolt_cellNumber_10000_replicate_1.csv"), row.names=1)
  uniqueCells <- sort(unique(c(multiome500$cell_type, multiome10000$cell_type, CITE500$cell_type, CITE10000$cell_type)))
  
  cols = gg_color_hue(length(uniqueCells))
  names(cols) <- uniqueCells
  
  cols
  #colorMap <- data.frame(cell_type = uniqueCells, cols = cols)
  #colorMap
}

getUMAPdf <- function(df) {
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
  umap_df
}

getGGplot <- function(csv.file, pointsize) {
  df <- read.csv(csv.file, row.names=1)
  umap_df <- getUMAPdf(df)
  gg <- ggplot2::ggplot(umap_df, 
                        aes(x = UMAP1, y = UMAP2, color = cell_type))+
    ggplot2::geom_point(alpha = 0.4, size=pointsize, shape=16)+
    ggplot2::labs(x = "UMAP1",
                  y = "UMAP2") +
    ggplot2::theme_bw() +
    ggplot2::theme(legend.title=element_blank(), legend.position="none") 

  gg <- gg + guides(color = guide_legend(override.aes = list(size=3)))
  gg
}

getGGplotPlusLegend <- function(csv.file, pointsize) {
  cols <- getColors()
  df <- read.csv(csv.file, row.names=1)
  umap_df <- getUMAPdf(df)
  gg.wLegend <- ggplot2::ggplot(umap_df, 
                                aes(x = UMAP1, y = UMAP2, color = cell_type))+
    ggplot2::geom_point(alpha = 0.4, size=pointsize, shape=16)+
    ggplot2::labs(x = "UMAP1",
                  y = "UMAP2") +
    ggplot2::scale_color_manual(values = cols, limits = force) +
    ggplot2::theme_bw() +
    ggplot2::theme(legend.title=element_blank(), legend.direction="horizontal")  +
    guides(color = guide_legend(override.aes = list(size=3)))
  
  legend <- cowplot::get_legend(gg.wLegend)
  
  gg.woLegend <- gg.wLegend + ggplot2::theme(legend.position="none") 
  
  list(gg.wLegend = gg.wLegend, 
       gg.woLegend = gg.woLegend, 
       legend = legend)
}



#setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
setwd("/Users/admin/Desktop/PhD/multi_omics_review_old/latent_and_meta")
#csv.files = list.files(path = "/Users/admin/Desktop/PhD/multi_omics_review/latent_and_meta", pattern = "csv$", full.names = TRUE)

toolsCITE <- c("Cobolt", "scMM", "totalvi", "SCALEX")
dataTypes <- c("Multiome", "CITE")


for (dataType in dataTypes){
  if (dataType == "CITE"){
    tools <- c("Cobolt", "scMM", "totalvi", "SCALEX")
  } else if (dataType == "Multiome"){
    tools <- c("Cobolt", "scMM", "multivi", "scMVP", "DAVAE", "portal")
  }
  
  lapply(tools, function(tool){
    print(tool)
    
    for (cellNumber in c("500", "10000")){
      csv.file <- paste0("/Users/admin/Desktop/PhD/multi_omics_review_old/latent_and_meta/dataType_", dataType, "_tool_", tool, "_cellNumber_", cellNumber,"_replicate_1.csv")
      if (cellNumber == "500"){
        pointsize <- 0.8
      } else if (cellNumber == "10000"){
        pointsize <- 0.4
      }
      # gg <- getGGplot(csv_path, pointsize = pointsize)
      # ggsave(paste0("umap_", dataType, "_", tool, "_", cellNumber, ".pdf"), gg,
      #        width = 4, height = 4)
      
      ggplots <- getGGplotPlusLegend(csv.file, pointsize = pointsize)
      ggsave(paste0("umap_", dataType, "_", tool, "_", cellNumber, "_woLegend.pdf"), ggplots[["gg.woLegend"]],
             width = 4, height = 4)
      
      ggsave(paste0("umap_", dataType, "_", tool, "_", cellNumber, "_wLegend.pdf"), ggplots[["gg.wLegend"]],
             width = 16, height = 8)
      
      if (dataType == "Multiome"){
       height <- 1.5
      } else {
        height <- 2.5
      }
      ggsave(paste0("umap_", dataType, "_", tool, "_", cellNumber, "_legend.pdf"), ggplots[["legend"]],
             width = 10, height = height)

    }
  })
}

session <- sessionInfo()
sink("umap_main_sessionInfo.txt")
print(session)
sink()
