library(rlist)
library(ggplot2)
library(dplyr)

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

tools <- list.dirs(getwd(),full.names = FALSE, 
                      recursive = FALSE) 
tools <- sort(setdiff(tools, "scMM_scMVP_DAVAE_Portal"))
toolTimeTables <- list()
for (tool in tools){
  print(tool)
  # list all txt files in folder tool, not recursive
  txtFiles <- list.files(tool, pattern='.txt')

  for (txtFile in txtFiles){
    print(txtFile)
    # unselect txt files containg "average" in file name
    if (!grepl("average", tolower(txtFile))){
      if (grepl("cite", tolower(txtFile))){
        mode <- "CITE"
      } else {
        mode <- "Multiome"
      }
      
      if (tool %in% c("Cobolt", "SCALEX")){
        sep <- ":"
        times.table <- read.table(paste0(tool, "/", txtFile), sep=sep)
        colNames <- c("nCells", "cpuTime")
      } else if (tool %in% c("multivi", "totalvi")){
        sep <- "\t"
        colNames <- c("cpuTime", "nCells")
      } else if (tool %in% c("DAVAE", "portal", "scMM", "scMVP")) {
        sep <- ","
        colNames <- c("nCells", "cpuTime")
      }

      times.table <- read.table(paste0(tool, "/", txtFile), sep=sep)
      colnames(times.table) <- colNames
      
      if (tool %in% c("Cobolt", "SCALEX")){
        times.table$nCells <- gsub(".*subsample_(.+)_cells.*", "\\1", times.table$nCells)
      }
      
      times.table[] <- lapply(times.table, as.numeric)
      times.table$tool <- tool
      times.table$mode <- mode
      times.table <- times.table[, c("tool", "mode", "nCells", "cpuTime")]
      times.table <- times.table[order(times.table$nCells),]
      
      toolTimeTables <- list.append(toolTimeTables, times.table)
    }
  }
}
names(toolTimeTables) <- tools
toolTimeTables.df <- do.call(rbind, toolTimeTables)

toolTimeTables.df.average <- toolTimeTables.df %>% dplyr::group_by(tool, mode, nCells) %>%
  dplyr::summarize(Mean = mean(cpuTime, na.rm=TRUE))

write.csv(toolTimeTables.df, "toolTimeTables.csv", row.names = FALSE)
write.csv(toolTimeTables.df.average, "toolTimeTablesMean.csv", row.names = FALSE)



toolTimeTables.df$tool_mode <- paste0(toolTimeTables.df$tool, " ", toolTimeTables.df$mode)

gg.ncells.cpu <- ggplot(toolTimeTables.df, aes(x=nCells, y=cpuTime, color=tool_mode, fill=tool_mode)) +
  geom_point(size=1, alpha=0.5, position = "jitter")+
  #stat_smooth(method = "lm")
  stat_summary(fun.data = "mean_cl_normal", 
               geom = "ribbon",
               fun.args=list(conf.int=0.95),
               alpha = 0.5) +
  stat_summary(fun.y = mean, geom = "line") +
  scale_x_continuous(breaks=c(500, 1000, 2500, 5000, 10000)) +
  xlab("Number of Cells") +
  ylab("CPU Time") +
  theme_bw() +
  theme(legend.position = "bottom", legend.title=element_blank())

ggsave("ncell_cpu.pdf", gg.ncells.cpu)

# ggplot(toolTimeTables.df, aes(x=log2(nCells), y=cpuTime, color=tool_mode, fill=tool_mode)) +
#   geom_point(size=1, alpha=0.5, position = "jitter") +
#   #stat_smooth(method = "lm")
#   stat_summary(fun.data = "mean_cl_normal", 
#                geom = "ribbon",
#                fun.args=list(conf.int=0.95),
#                alpha = 0.5) +
#   stat_summary(fun.y = mean, geom = "line") +
#   #scale_x_continuous(breaks=c(500, 1000, 2500, 5000, 10000)) +
#   xlab("Log2(Number of Cells)") +
#   ylab("CPU Time") +
#   theme_bw() +
#   theme(legend.position = "bottom", legend.title=element_blank())

gg.ncells.log2cpu <- ggplot(toolTimeTables.df, aes(x=nCells, y=log2(cpuTime), color=tool_mode, fill=tool_mode)) +
  geom_point(size=1, alpha=0.5, position = "jitter")+
  #stat_smooth(method = "lm")
  stat_summary(fun.data = "mean_cl_normal", 
               geom = "ribbon",
               fun.args=list(conf.int=0.95),
               alpha = 0.5) + 
  stat_summary(fun.y = mean, geom = "line") +
  scale_x_continuous(breaks=c(500, 1000, 2500, 5000, 10000)) +
  xlab("Number of Cells") +
  ylab("Log2(CPU Time)") +
  theme_bw() +
  theme(legend.position = "bottom", legend.title=element_blank())

ggsave("ncell_log2cpu.pdf", gg.ncells.log2cpu)


gg.log2ncells.log2cpu <- ggplot(toolTimeTables.df, aes(x=log2(nCells), y=log2(cpuTime), color=tool_mode, fill=tool_mode)) +
  geom_point(size=1, alpha=0.5, position = "jitter") +
  #stat_smooth(method = "lm")
  stat_summary(fun.data = "mean_cl_normal", 
               geom = "ribbon",
               fun.args=list(conf.int=0.95),
               alpha = 0.5) +
  stat_summary(fun.y = mean, geom = "line") +
  #scale_x_continuous(breaks=c(500, 1000, 2500, 5000, 10000)) +
  xlab("Log2(Number of Cells)") +
  ylab("Log2(CPU Time)") +
  theme_bw() +
  theme(legend.position = "bottom", legend.title=element_blank())

ggsave("log2ncell_log2cpu.pdf", gg.log2ncells.log2cpu)

session <- sessionInfo()
sink("nCellsVsCPU_sessionInfo.txt")
print(session)
sink()
