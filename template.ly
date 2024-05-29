\version "2.18.2"
\paper {
  #(set-paper-size "a4")
  indent = 0\mm
  line-width = 200\mm
  top-margin = 10\mm
  bottom-margin = 10\mm
  left-margin = 10\mm
  right-margin = 10\mm
}
\layout {
  \context {
    \Staff
    \remove "Time_signature_engraver"
  }
}
