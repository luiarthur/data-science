shinyServer(
  function(input, output, session) {
    output$myPlot = renderPlot({
      distribution = input$Distribution
      sample_size = input$SampleSize
      samples = NULL
      if (distribution == "Normal") {
        samples = rnorm(sample_size, as.numeric(input$Mean), as.numeric(input$SD))
      } else if (distribution == "Expo") {
        samples = rexp(sample_size, rate=as.numeric(input$rate))
      }
      hist(samples)
    })
  }
)
