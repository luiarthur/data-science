shinyUI(
  pageWithSidebar(
    headerPanel("My First"),
    sidebarPanel(
      selectInput("Distribution", "Please Select", choices=c('Normal', 'Expo')),
      sliderInput('SampleSize', 'Select Sample Size: ', min=10, max=10000, value=10),
      conditionalPanel(
        condition="input.Distribution == 'Normal'",
        textInput('Mean', 'Input Mean', 0),
        textInput('SD', 'Input SD', 1)
      ),
      conditionalPanel(
        condition="input.Distribution == 'Expo'",
        textInput('rate', 'Input rate', 1)
      )
    ),
    mainPanel(
      plotOutput("myPlot")
    )
  )
)
