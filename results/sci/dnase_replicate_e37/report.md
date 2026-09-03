# E37 replicate-curve SCI on the DNase assay

The analysis used 11 assay runs, eight concentrations, and all 176 raw
measurements. Technical duplicates were averaged inside each run. The run,
not the individual optical-density reading, was the independent sampling unit.

The 95% SCI set for the mean curve's convex-to-concave transition is
`[0.78125, 12.5]` in concentration units. It contains a lower bound but reaches the largest observed concentration. Thus the data support that the transition does not occur below `0.78125`, but they do not give a reliable upper bound within the observed range.

A four-parameter logistic fit places its descriptive transition at
`4.1412`. This point estimate lies inside the SCI set, but
it does not have the same finite-sample guarantee.

The Student construction allows arbitrary dependence and unequal variance
between concentrations inside an assay run. Its exact coverage statement
assumes that the 11 run-level curves are independent Gaussian replicates with
a common mean and covariance.
