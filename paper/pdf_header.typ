// PDF layout fixes: breakable tables (no floats), tighter spacing, readable tables.

#show figure.where(kind: table): it => block(breakable: true, width: 100%, above: 0.55em, below: 0.55em)[
  #set text(size: 9pt, hyphenate: false)
  #it.body
]

#show figure.where(kind: image): it => block(breakable: true, width: 100%, above: 0.75em, below: 0.75em)[
  #align(center)[#it.body]
]

#show heading: set block(breakable: true, above: 1.05em, below: 0.45em)
#show heading: set heading(outlined: true, bookmarked: true)

#set table(
  inset: 4pt,
  stroke: 0.3pt + luma(210),
  fill: (x, y) => if y == 0 { luma(248) },
)
