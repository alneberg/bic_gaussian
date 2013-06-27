#!/usr/bin/env python
import fileinput
import sys
import os
from argparse import ArgumentParser

import itertools
from scipy import linalg
import pylab as pl
import matplotlib as mpl


import pandas as p
import sklearn.mixture as mixture
import numpy as np

def main(coverage_file,output_base,header=None):
    df = p.read_csv(coverage_file,header=header)
    X = df.ix[:,1:].values


    lowest_bic = np.infty
    bic = []
    convergence = []
    settings = []
    n_components_range = [1,2,3,5,7]
    possible_cv_types = ['spherical', 'tied', 'diag', 'full']
    cv_types = ['spherical','tied','diag','full']
    for cv_type in cv_types:
        for n_components in n_components_range:
            # Fit a mixture of gaussians with EM
            gmm = mixture.GMM(n_components=n_components, covariance_type=cv_type)
            gmm.fit(X)
            labels = gmm.predict(X)
            bic.append(gmm.bic(X))
            convergence.append(gmm.converged_)
            class_series = p.Series(labels,index=df[0].values)
            setting = str(n_components)+"_" + str(cv_type)
            settings.append(setting)
            class_series.to_csv(output_base+"_" +setting+str(gmm.bic(X))+ '.csv')
            if bic[-1] < lowest_bic:
                lowest_bic = bic[-1]
                best_gmm = gmm
    
    bic_series = p.Series(bic,index=settings)
    convergence_series = p.Series(convergence,index=settings)
    bic_df = p.DataFrame({'bic': bic_series, 'converged': convergence_series})
    bic_df.to_csv(output_base+"_BIC_results.csv")
    bic = np.array(bic)
    color_iter = itertools.cycle(['k', 'r', 'g', 'b', 'c', 'm', 'y'])
    clf = best_gmm
    bars = []
    # Plot the BIC scores
    spl = pl.subplot(1, 1, 1)
    for i, (cv_type, color) in enumerate(zip(cv_types, color_iter)):
        xpos = np.array(n_components_range) + .2 * (i - 2)
        bars.append(pl.bar(xpos, bic[i * len(n_components_range):
                                         (i + 1) * len(n_components_range)],
                           width=.2, color=color))
    pl.xticks(n_components_range)
    pl.ylim([bic.min() * 1.01 - .01 * bic.max(), bic.max()])
    pl.title('BIC score per model')
    xpos = np.mod(bic.argmin(), len(n_components_range)) + .65 +\
        .2 * np.floor(bic.argmin() / len(n_components_range))
    pl.text(xpos, bic.min() * 0.97 + .03 * bic.max(), '*', fontsize=14)
    spl.set_xlabel('Number of components')
    spl.legend([b[0] for b in bars], cv_types)
    pl.savefig(output_base+"_bic_comparison.png")


if __name__=="__main__":
    parser = ArgumentParser()
    parser.add_argument('coverage',
                        help='specify the coverage file')
    parser.add_argument('-o','--output',
                        help='specify the output base file_name, the number of components and cv_type will be added to this file name.')
    parser.add_argument('--header', action='store_true',
                        help='Use this tag if header is included in coverage file')
    args = parser.parse_args()
    if not args.output:
       sys.exit(-1) 
    if args.header:
        header=0
    else:
        header=None
    main(args.coverage,args.output,header)

