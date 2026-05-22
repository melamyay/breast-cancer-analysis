# =============================================================================
# SHINY APP : Breast Cancer Drug Sensitivity Analysis
# =============================================================================
# install.packages(c("shiny", "shinydashboard", "plotly", "DT",
#                    "dplyr", "tidyr", "RColorBrewer", "shinyWidgets"))
# =============================================================================

library(shiny)
library(shinydashboard)
library(plotly)
library(DT)
library(dplyr)
library(tidyr)
library(shinyWidgets)

# =============================================================================
# DONNÉES
# =============================================================================

DATA_DIR   <- "Data/"
OUTPUT_DIR <- "Output/"

drug_df       <- read.csv(paste0(OUTPUT_DIR, "breast_drug_sensitivity_clean.csv"), check.names = FALSE)
subtype_means <- read.csv(paste0(OUTPUT_DIR, "subtype_drug_means.csv"), check.names = FALSE, row.names = 1)
classif_df    <- read.csv(paste0(DATA_DIR, "classification_uploadedData.csv"))
kw_df         <- read.csv(paste0(OUTPUT_DIR, "kruskal_wallis_results.csv"))
sig_drugs     <- read.csv(paste0(OUTPUT_DIR, "significant_drugs.csv"))
ml_results    <- read.csv(paste0(OUTPUT_DIR, "ml_regression_results.csv"))
posthoc_df    <- read.csv(paste0(OUTPUT_DIR, "posthoc_mannwhitney_results.csv"))

classif_df <- classif_df %>%
  rename(Cell_Line = Cell.lines, Subtype = cluster) %>%
  select(Cell_Line, Subtype, maxProb)

cell_line_col <- colnames(drug_df)[1]
merged_df <- drug_df %>%
  left_join(classif_df, by = setNames("Cell_Line", cell_line_col))

drug_matrix <- drug_df %>% select(-1)
drug_matrix[is.na(drug_matrix)] <- colMeans(drug_matrix, na.rm = TRUE)[col(drug_matrix)[is.na(drug_matrix)]]
drug_matrix <- drug_matrix[, apply(drug_matrix, 2, var) > 0]
pca_result  <- prcomp(scale(drug_matrix))
pca_df <- data.frame(
  Cell_Line = drug_df[[1]],
  PCA1      = pca_result$x[, 1],
  PCA2      = pca_result$x[, 2]
) %>% left_join(classif_df, by = "Cell_Line")

short_name      <- function(x) sub(" \\(.*", "", x)
kw_df$ShortName <- short_name(kw_df$Drug)
sig_drugs$ShortName <- short_name(sig_drugs$Drug)

subtypes_all <- sort(unique(classif_df$Subtype))
PALETTE <- c("HER2-enriched"       = "#66C2A5",
             "Luminal"             = "#FC8D62",
             "Luminal-infiltrated" = "#8DA0CB",
             "Normal-like"         = "#E78AC3",
             "TNBC-Basal"          = "#A6D854",
             "TNBC-Mes"            = "#FFD92F")

CSS <- "
  .content-wrapper, .right-side { background-color: #0f1117 !important; }
  .main-header .logo, .main-header .navbar { background-color: #1a1d2e !important; border-bottom: 1px solid #e91e8c; }
  .main-sidebar { background-color: #1a1d2e !important; }
  .sidebar-menu > li > a { color: #a0a8c0 !important; font-size: 13px; }
  .sidebar-menu > li.active > a,
  .sidebar-menu > li > a:hover { color: #fff !important; background-color: #e91e8c22 !important; border-left: 3px solid #e91e8c; }
  .box { background-color: #1a1d2e !important; border: 1px solid #2a2d3e !important; border-radius: 12px !important; color: #e0e0e0 !important; }
  .box-header { background-color: transparent !important; color: #fff !important; border-bottom: 1px solid #2a2d3e !important; }
  .box.box-solid > .box-header { border-radius: 12px 12px 0 0 !important; }
  .box.box-primary.box-solid > .box-header { background-color: #1e3a5f !important; }
  .box.box-success.box-solid  > .box-header { background-color: #1a3a2a !important; }
  .box.box-warning.box-solid  > .box-header { background-color: #3a2a0a !important; }
  .box.box-danger.box-solid   > .box-header { background-color: #3a1a1a !important; }
  .box.box-info.box-solid     > .box-header { background-color: #1a2a3a !important; }
  .info-box { border-radius: 10px !important; background-color: #1a1d2e !important; border: 1px solid #2a2d3e; }
  .info-box-text, .info-box-number { color: #fff !important; }
  .form-control, .selectize-input { background-color: #0f1117 !important; border: 1px solid #2a2d3e !important; color: #e0e0e0 !important; border-radius: 8px !important; }
  .selectize-dropdown { background-color: #1a1d2e !important; color: #e0e0e0 !important; }
  .selectize-dropdown .option:hover { background-color: #e91e8c33 !important; }
  .checkbox label, .radio label { color: #a0a8c0 !important; }
  .irs-bar, .irs-bar-edge { background-color: #e91e8c !important; border-color: #e91e8c !important; }
  .irs-single { background-color: #e91e8c !important; }
  .dataTables_wrapper { color: #e0e0e0 !important; }
  table.dataTable { background-color: #1a1d2e !important; color: #e0e0e0 !important; }
  table.dataTable thead th { background-color: #0f1117 !important; color: #e91e8c !important; border-bottom: 1px solid #2a2d3e !important; }
  table.dataTable tbody tr:hover { background-color: #e91e8c22 !important; }
  .dataTables_paginate .paginate_button { color: #a0a8c0 !important; }
  .dataTables_paginate .paginate_button.current { background: #e91e8c !important; color: white !important; border-radius: 5px; }
  ::-webkit-scrollbar { width: 6px; }
  ::-webkit-scrollbar-track { background: #0f1117; }
  ::-webkit-scrollbar-thumb { background: #e91e8c; border-radius: 3px; }
  h4 { color: #e91e8c !important; }
  p, li { color: #a0a8c0; }
  strong { color: #fff; }
  em { color: #7a8099; }
  hr { border-color: #2a2d3e; }
"

# =============================================================================
# UI
# =============================================================================

ui <- dashboardPage(
  skin = "blue",
  
  dashboardHeader(title = "Breast Cancer Drug Analysis"),
  
  dashboardSidebar(
    sidebarMenu(
      menuItem("Vue d'ensemble",   tabName = "overview",  icon = icon("chart-pie")),
      menuItem("PCA",              tabName = "pca",       icon = icon("circle-dot")),
      menuItem("Top médicaments",  tabName = "topdrugs",  icon = icon("pills")),
      menuItem("Heatmap",          tabName = "heatmap",   icon = icon("table-cells")),
      menuItem("Boxplots",         tabName = "boxplots",  icon = icon("box")),
      menuItem("Tests stat.",      tabName = "stats",     icon = icon("flask")),
      menuItem("Machine Learning", tabName = "ml",        icon = icon("robot"))
    )
  ),
  
  dashboardBody(
    tags$head(tags$style(HTML(CSS))),
    
    tabItems(
      
      # 1. VUE D'ENSEMBLE -------------------------------------------------------
      tabItem(tabName = "overview",
              fluidRow(
                infoBox("Lignées analysées",  nrow(drug_df),       icon = icon("dna"),        color = "blue"),
                infoBox("Médicaments",        ncol(drug_df) - 1,   icon = icon("pills"),      color = "green"),
                infoBox("Médicaments sign.",  nrow(sig_drugs),     icon = icon("star"),       color = "orange")
              ),
              fluidRow(
                box(title = "Distribution des sous-types moléculaires",
                    width = 6, status = "primary", solidHeader = TRUE,
                    plotlyOutput("overview_subtype_plot", height = 350)),
                box(title = "Top 15 médicaments les plus significatifs (Kruskal-Wallis)",
                    width = 6, status = "warning", solidHeader = TRUE,
                    plotlyOutput("overview_sig_drugs", height = 350))
              ),
              fluidRow(
                box(title = "À propos", width = 12, status = "info",
                    p("Analyse de la sensibilité aux médicaments (PRISM Drug Screen — DepMap)
                pour les lignées de cancer du sein, classifiées en sous-types moléculaires
                via ", strong("cRegMap"), " (brcaregmap)."),
                    p("Pipeline : filtrage DepMap → classification cRegMap → tests statistiques
                (Kruskal-Wallis + correction FDR Benjamini-Hochberg) → Machine Learning (Random Forest, ElasticNet, Régression linéaire)."),
                    tags$ul(
                      tags$li("30 lignées breast cancer | 1360 médicaments"),
                      tags$li("4 sous-types identifiés par cRegMap — 3 retenus pour les tests (Luminal, TNBC-Basal, TNBC-Mes) — HER2 exclu car n=2"),
                      tags$li("35 médicaments avec p < 0.01 (exploratoire — aucun ne survit au FDR q < 0.05)"),
                      tags$li("Classification ML : 85.9% accuracy (régression logistique, LOO CV)"),
                      tags$li("Meilleur R² régression AUC : 0.494 (Idazoxan, Régression linéaire, LOO CV)")
                    ))
              )
      ),
      
      # 2. PCA ------------------------------------------------------------------
      tabItem(tabName = "pca",
              fluidRow(
                box(title = "Options", width = 3, status = "primary", solidHeader = TRUE,
                    checkboxGroupInput("pca_subtypes", "Sous-types à afficher :",
                                       choices = subtypes_all, selected = subtypes_all),
                    hr(),
                    sliderInput("pca_point_size", "Taille des points :", 5, 20, 10),
                    checkboxInput("pca_labels", "Afficher les noms des lignées", FALSE)),
                box(title = "PCA – Profil pharmacologique coloré par sous-type moléculaire",
                    width = 9, status = "primary", solidHeader = TRUE,
                    plotlyOutput("pca_plot", height = 550))
              )
      ),
      
      # 3. TOP MÉDICAMENTS ------------------------------------------------------
      tabItem(tabName = "topdrugs",
              fluidRow(
                box(title = "Options", width = 3, status = "success", solidHeader = TRUE,
                    selectInput("top_subtype", "Sous-type :", choices = rownames(subtype_means)),
                    sliderInput("top_n", "Nombre de médicaments :", 5, 30, 10),
                    hr(),
                    p(em("AUC basse = médicament plus efficace"))),
                box(title = "Top médicaments par sous-type moléculaire",
                    width = 9, status = "success", solidHeader = TRUE,
                    plotlyOutput("top_drugs_plot", height = 500))
              ),
              fluidRow(
                box(title = "Tableau des AUC moyennes", width = 12,
                    status = "success", solidHeader = TRUE,
                    DTOutput("top_drugs_table"))
              )
      ),
      
      # 4. HEATMAP --------------------------------------------------------------
      tabItem(tabName = "heatmap",
              fluidRow(
                box(title = "Options", width = 3, status = "warning", solidHeader = TRUE,
                    sliderInput("heatmap_top_n", "Top N médicaments par sous-type :", 5, 20, 10),
                    checkboxGroupInput("heatmap_subtypes", "Sous-types :",
                                       choices  = rownames(subtype_means),
                                       selected = rownames(subtype_means)),
                    hr(),
                    p(em("Rouge = AUC basse = plus efficace"))),
                box(title = "Heatmap – AUC moyennes des meilleurs médicaments",
                    width = 9, status = "warning", solidHeader = TRUE,
                    plotlyOutput("heatmap_plot", height = 550))
              )
      ),
      
      # 5. BOXPLOTS -------------------------------------------------------------
      tabItem(tabName = "boxplots",
              fluidRow(
                box(title = "Options", width = 3, status = "danger", solidHeader = TRUE,
                    selectizeInput("boxplot_drug", "Médicament :",
                                   choices  = short_name(colnames(drug_df)[-1]),
                                   selected = sig_drugs$ShortName[1],
                                   options  = list(placeholder = "Rechercher...")),
                    checkboxGroupInput("boxplot_subtypes", "Sous-types :",
                                       choices  = subtypes_all,
                                       selected = c("Luminal", "TNBC-Basal", "TNBC-Mes")),
                    hr(),
                    uiOutput("boxplot_kw_result")),
                box(title = "Distribution des AUC par sous-type",
                    width = 9, status = "danger", solidHeader = TRUE,
                    plotlyOutput("boxplot_plot", height = 500))
              )
      ),
      
      # 6. TESTS STAT -----------------------------------------------------------
      tabItem(tabName = "stats",
              fluidRow(
                box(title = "Médicaments top 35 (p brute < 0.01, exploratoire — FDR q < 0.05 : aucun significatif)",
                    width = 12, status = "info", solidHeader = TRUE,
                    DTOutput("sig_drugs_table"))
              ),
              fluidRow(
                box(title = "Volcano plot – Significance vs Dispersion inter-sous-type",
                    width = 8, status = "info", solidHeader = TRUE,
                    plotlyOutput("volcano_plot", height = 500)),
                box(title = "Comparaisons post-hoc (Mann-Whitney, p < 0.05)",
                    width = 4, status = "info", solidHeader = TRUE,
                    DTOutput("posthoc_table"))
              )
      ),
      
      # 7. ML -------------------------------------------------------------------
      tabItem(tabName = "ml",
              fluidRow(
                infoBox("Accuracy RF (sous-type)",    "84.5%", icon = icon("robot"),      color = "green"),
                infoBox("Accuracy Log. (sous-type)",  "85.9%", icon = icon("brain"),      color = "blue"),
                infoBox("Meilleur R² (Idazoxan rég. lin.)", "0.494", icon = icon("chart-line"), color = "orange")
              ),
              fluidRow(
                box(title = "R² par médicament et modèle (LOO CV)",
                    width = 7, status = "primary", solidHeader = TRUE,
                    plotlyOutput("ml_heatmap", height = 400)),
                box(title = "Interprétation", width = 5, status = "primary", solidHeader = TRUE,
                    h4("Partie A — Classification du sous-type"),
                    p("Les scores TF (440 régulateurs) permettent de prédire le sous-type
                moléculaire avec ", strong("85.9% d'accuracy"), " (LOO cross-validation).
                Cela confirme que la signature transcriptomique cRegMap capture bien
                l'identité biologique des sous-types."),
                    hr(),
                    h4("Partie B — Prédiction de l'AUC"),
                    p("Méthode : LOO cross-validation + PCA 15 composantes (ElasticNet, Régression lin.).
                ", strong("IDAZOXAN"), " montre le meilleur R² global (R² = 0.494, Régression linéaire).
                ", strong("LY2603618"), " atteint R² = 0.413 en Random Forest — doublement intéressant
                car c'est aussi le meilleur signal parmi les inhibiteurs de checkpoint ATR/CHK1,
                convergence entre stats et ML."),
                    hr(),
                    p(em("Validation : LOO pour classification et régression | PCA 15 composantes pour modèles linéaires")))
              ),
              fluidRow(
                box(title = "Résultats détaillés", width = 12,
                    status = "primary", solidHeader = TRUE,
                    DTOutput("ml_table"))
              )
      )
    ) # fin tabItems
  )   # fin dashboardBody
)     # fin dashboardPage


# =============================================================================
# SERVER
# =============================================================================

server <- function(input, output, session) {
  
  # Overview ------------------------------------------------------------------
  output$overview_subtype_plot <- renderPlotly({
    counts <- classif_df %>% count(Subtype) %>% arrange(desc(n))
    plot_ly(counts, x = ~Subtype, y = ~n, type = "bar",
            marker = list(color = PALETTE[counts$Subtype],
                          line = list(color = "white", width = 1))) %>%
      layout(xaxis = list(title = ""), yaxis = list(title = "Nombre de lignées"),
             plot_bgcolor = "rgba(0,0,0,0)", paper_bgcolor = "rgba(0,0,0,0)",
             font = list(color = "#e0e0e0"))
  })
  
  output$overview_sig_drugs <- renderPlotly({
    top15 <- sig_drugs %>% arrange(p_value) %>% head(15)
    plot_ly(top15, x = ~p_value, y = ~reorder(ShortName, -p_value),
            type = "bar", orientation = "h",
            marker = list(color = "#e91e8c")) %>%
      layout(xaxis = list(title = "p-value"), yaxis = list(title = ""),
             plot_bgcolor = "rgba(0,0,0,0)", paper_bgcolor = "rgba(0,0,0,0)",
             font = list(color = "#e0e0e0"))
  })
  
  # PCA -----------------------------------------------------------------------
  output$pca_plot <- renderPlotly({
    df <- pca_df %>% filter(Subtype %in% input$pca_subtypes)
    plot_ly(df, x = ~PCA1, y = ~PCA2, color = ~Subtype,
            colors = PALETTE, type = "scatter", mode = "markers",
            text = ~Cell_Line,
            hovertemplate = if (input$pca_labels)
              "%{text}<br>PC1=%{x:.2f} PC2=%{y:.2f}<extra></extra>"
            else "%{text}<extra></extra>",
            marker = list(size = input$pca_point_size,
                          line = list(color = "white", width = 1))) %>%
      layout(xaxis = list(title = "Composante principale 1"),
             yaxis = list(title = "Composante principale 2"),
             legend = list(title = list(text = "Sous-type")),
             plot_bgcolor = "rgba(0,0,0,0)", paper_bgcolor = "rgba(0,0,0,0)",
             font = list(color = "#e0e0e0"))
  })
  
  # Top médicaments -----------------------------------------------------------
  top_drugs_data <- reactive({
    subtype_means[input$top_subtype, ] %>%
      t() %>% as.data.frame() %>%
      setNames("AUC") %>%
      mutate(Drug = short_name(rownames(.))) %>%
      arrange(AUC) %>%
      head(input$top_n)
  })
  
  output$top_drugs_plot <- renderPlotly({
    df    <- top_drugs_data()
    color <- PALETTE[input$top_subtype]
    if (is.na(color)) color <- "#e91e8c"
    plot_ly(df, x = ~AUC, y = ~reorder(Drug, -AUC),
            type = "bar", orientation = "h",
            marker = list(color = color, line = list(color = "white", width = 0.5))) %>%
      add_text(text = ~round(AUC, 3), textposition = "outside",
               textfont = list(color = "#e0e0e0")) %>%
      layout(xaxis = list(title = "AUC moyenne (↓ = plus efficace)"),
             yaxis = list(title = ""), showlegend = FALSE,
             plot_bgcolor = "rgba(0,0,0,0)", paper_bgcolor = "rgba(0,0,0,0)",
             font = list(color = "#e0e0e0"))
  })
  
  output$top_drugs_table <- renderDT({
    df <- subtype_means %>%
      t() %>% as.data.frame() %>%
      mutate(Drug = short_name(rownames(.))) %>%
      select(Drug, everything()) %>%
      arrange(.data[[input$top_subtype]]) %>%
      head(input$top_n)
    datatable(df, options = list(pageLength = 10, scrollX = TRUE), rownames = FALSE) %>%
      formatRound(columns = rownames(subtype_means), digits = 4)
  })
  
  # Heatmap -------------------------------------------------------------------
  output$heatmap_plot <- renderPlotly({
    req(length(input$heatmap_subtypes) > 0)
    sub_sel   <- input$heatmap_subtypes
    sm        <- subtype_means[sub_sel, , drop = FALSE]
    top_drugs <- unique(unlist(lapply(sub_sel, function(s) {
      order(as.numeric(sm[s, ]))[1:input$heatmap_top_n]
    })))
    mat           <- sm[, top_drugs, drop = FALSE]
    colnames(mat) <- short_name(colnames(mat))
    plot_ly(x = colnames(mat), y = rownames(mat), z = as.matrix(mat),
            type = "heatmap", colorscale = "RdYlGn", reversescale = TRUE,
            colorbar = list(title = "AUC moyenne"),
            hovertemplate = "%{y} — %{x}<br>AUC = %{z:.3f}<extra></extra>") %>%
      layout(xaxis = list(tickangle = -40, title = "Médicaments"),
             yaxis = list(title = "Sous-type"),
             plot_bgcolor = "rgba(0,0,0,0)", paper_bgcolor = "rgba(0,0,0,0)",
             font = list(color = "#e0e0e0"))
  })
  
  # Boxplots ------------------------------------------------------------------
  output$boxplot_kw_result <- renderUI({
    drug_full <- colnames(drug_df)[-1][short_name(colnames(drug_df)[-1]) == input$boxplot_drug]
    if (length(drug_full) == 0) return(NULL)
    kw_row <- kw_df %>% filter(Drug == drug_full[1])
    if (nrow(kw_row) == 0) return(p("Non testé"))
    p_val <- kw_row$p_value[1]
    col   <- if (p_val < 0.01) "#2ecc71" else "#a0a8c0"
    tags$p(style = paste0("color:", col, ";font-weight:bold"),
           sprintf("KW p = %.4f %s", p_val, if (p_val < 0.01) "✅" else ""))
  })
  
  output$boxplot_plot <- renderPlotly({
    drug_full <- colnames(drug_df)[-1][short_name(colnames(drug_df)[-1]) == input$boxplot_drug]
    req(length(drug_full) > 0)
    df <- merged_df %>%
      filter(Subtype %in% input$boxplot_subtypes) %>%
      select(Subtype, AUC = all_of(drug_full[1])) %>%
      filter(!is.na(AUC))
    plot_ly(df, x = ~Subtype, y = ~AUC, color = ~Subtype,
            colors = PALETTE, type = "box",
            boxpoints = "all", jitter = 0.3, pointpos = 0,
            marker = list(size = 5, opacity = 0.6)) %>%
      layout(xaxis = list(title = ""), yaxis = list(title = "AUC"),
             showlegend = FALSE,
             title = list(text = input$boxplot_drug, font = list(size = 14, color = "#fff")),
             plot_bgcolor = "rgba(0,0,0,0)", paper_bgcolor = "rgba(0,0,0,0)",
             font = list(color = "#e0e0e0"))
  })
  
  # Tests stat ----------------------------------------------------------------
  output$sig_drugs_table <- renderDT({
    df <- sig_drugs %>%
      select(ShortName, KW_stat, p_value, any_of("p_fdr")) %>%
      rename(Médicament = ShortName, `KW stat` = KW_stat, `p-value brute` = p_value)
    if ("p_fdr" %in% colnames(df)) df <- rename(df, `p-value FDR` = p_fdr)
    df %>%
      arrange(`p-value brute`) %>%
      datatable(options = list(pageLength = 15), rownames = FALSE) %>%
      formatRound(columns = intersect(c("KW stat", "p-value brute", "p-value FDR"), colnames(df)),
                  digits = 4)
  })
  
  output$volcano_plot <- renderPlotly({
    # Calcul dispersion : std des moyennes AUC par sous-type (comme Python)
    drug_cols <- colnames(drug_df)[-1]
    sub3      <- c("Luminal", "TNBC-Basal", "TNBC-Mes")
    means_mat <- subtype_means[sub3, drug_cols, drop = FALSE]
    effect_sizes <- apply(means_mat, 2, sd, na.rm = TRUE)
    
    df <- kw_df %>%
      mutate(
        log10p      = -log10(p_value),
        effect_size = effect_sizes[Drug],
        significant = if ("p_fdr" %in% colnames(kw_df)) p_fdr < 0.05 else FALSE
      )
    plot_ly(df, x = ~effect_size, y = ~log10p, color = ~significant,
            colors = c("FALSE" = "#a8c4e0", "TRUE" = "#e91e8c"),
            type = "scatter", mode = "markers", text = ~ShortName,
            hovertemplate = "%{text}<br>Dispersion=%{x:.3f}<br>-log10(p)=%{y:.2f}<extra></extra>",
            marker = list(size = 7, opacity = 0.6,
                          line = list(color = "white", width = 0.3))) %>%
      add_segments(x = 0, xend = max(df$effect_size, na.rm = TRUE),
                   y = -log10(0.01), yend = -log10(0.01),
                   line = list(dash = "dash", color = "#aaa", width = 1),
                   showlegend = FALSE) %>%
      layout(xaxis = list(title = "Dispersion inter-sous-type (std des moyennes AUC)"),
             yaxis = list(title = "-log10(p-value)"),
             legend = list(title = list(text = "FDR q < 0.05")),
             plot_bgcolor = "rgba(0,0,0,0)", paper_bgcolor = "rgba(0,0,0,0)",
             font = list(color = "#e0e0e0"))
  })
  
  output$posthoc_table <- renderDT({
    posthoc_df %>%
      filter(p_value < 0.05) %>%
      mutate(Drug = short_name(Drug)) %>%
      select(Drug, Subtype_A, Subtype_B, p_value, mean_AUC_A, mean_AUC_B) %>%
      arrange(p_value) %>%
      datatable(options = list(pageLength = 10, scrollX = TRUE), rownames = FALSE) %>%
      formatRound(columns = c("p_value", "mean_AUC_A", "mean_AUC_B"), digits = 4)
  })
  
  # ML ------------------------------------------------------------------------
  output$ml_heatmap <- renderPlotly({
    pivot <- ml_results %>%
      select(Drug, Modèle, R2_CV) %>%
      pivot_wider(names_from = Modèle, values_from = R2_CV)
    mat <- as.matrix(pivot[, -1])
    rownames(mat) <- pivot$Drug
    plot_ly(x = colnames(mat), y = rownames(mat), z = mat,
            type = "heatmap", colorscale = "RdYlGn", zmin = -1, zmax = 1,
            colorbar = list(title = "R² (LOO CV)"),
            hovertemplate = "%{y} — %{x}<br>R² = %{z:.3f}<extra></extra>") %>%
      layout(xaxis = list(title = ""), yaxis = list(title = ""),
             plot_bgcolor = "rgba(0,0,0,0)", paper_bgcolor = "rgba(0,0,0,0)",
             font = list(color = "#e0e0e0"))
  })
  
  output$ml_table <- renderDT({
    ml_results %>%
      arrange(Drug, desc(R2_CV)) %>%
      datatable(options = list(pageLength = 15), rownames = FALSE) %>%
      formatRound(columns = c("R2_CV", "RMSE_CV"), digits = 4)
  })
}

shinyApp(ui = ui, server = server)