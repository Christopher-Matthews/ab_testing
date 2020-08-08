import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import scipy.stats as stats
from statsmodels.stats.proportion import proportions_ztest as z_test
from statsmodels.stats.proportion import proportion_effectsize
from statsmodels.stats.power import tt_ind_solve_power



def show_available_functions():
    print(
        '''
        List of available functions inside ab_test_analysis.py:

        check_test_split: check test groups split properly
        std_statistical_significance: significance for binary outcomes
        find_conservative_test_size: find test group sizes
        power_graph: get graphical representation of Test Power
        bootstrap_point_estimator: Estimate Mean, Median, Quantile
        plot_bootstrap_dist_comparison: Show testing results from test
            and control groups
        ''')


def check_test_split(
    n_control:int,
    n_experiment:int,
    split_perc:float=0.50,
    n_trials:int=2_000_000) -> 'Probability Statement':

        '''
            Input:
                n_control: # of participants in control group
                n_experiment: participants in experiment group
                split_perc: expected % of partic in experiment group
                n_trials: number of simulations to run
            Output:
                simulation results statment
                estimated p-value: int
            Example:
                check_test_split(5000,4900)

                Running a simulation found that 16.12%
                of the test splits had samples sizes as
                extreme as our test.
                (Approx P-value: 0.1612105)
        '''

        n_obs = n_control + n_experiment
        var = np.abs(n_control - n_experiment)
        samples = np.random.binomial(n_obs, split_perc, n_trials)
        outliers = np.logical_or(samples >= n_control + var,
                                 samples <= n_control - var).sum()
        perc_outlier = round(outliers*100/n_trials, 2)
        print('Running a simulation found that {}% of '
              'the test splits had\nsamples sizes'
              ' as extreme as our test. (Approx P-value: {})'
              .format(perc_outlier, outliers/n_trials))

        return outliers/n_trials


def std_statistical_significance(
    n_control:int,
    n_experiment:int,
    control_convs:int,
    exper_convs:int,
    alternative:str='two-sided') -> 'significance statement':

    '''
        Input:
            n_control: # people in control group
            n_experiment: # people in experiment group
            control_convs: # binary conversion for control
            exper_convs: # binary conversions for experiment
            alternative: rejection region for test
        Output:
            pvalue statement
            tuple: (test statistic, p_value)
        Example:
            std_statistical_significance(5000,5000,258,308)
    '''

    stat, p_val = z_test(
        nobs=np.array([n_control, n_experiment]),
        count=np.array([control_convs, exper_convs]),
        alternative=alternative
    )
    print('P-Value: {}'.format(p_val))

    return stat, p_val


def find_conservative_test_size(
    est_control_conv:float,
    est_test_conv:float,
    alpha:float=0.025,
    power:float=0.80,
    alternative:str='two-sided',
    ratio=1) -> float:

    '''
    Input:
        est_control_conv: estimated conrol conversion metric
        est_test_conv: estimated test conversion metric
        alpha: size of rejection region
        power: test power desired
        alternative: default two-sided.
            Set to larger if you plan on your metric increasing
            Set to smaller if you plan on decreasing a metric
        ratio: nobs2 = nobs1 * ratio so 1 = 50-50 split
    Output:
        est size of control group. Double to get total audience size if
        50-50 test
    Example:
        find_test_size(.02, .025, alternative='larger') outputs
        13768.939916102516 - so you need a control group = 13,769
        and a test group of 13,769
    '''

    control_size = tt_ind_solve_power(
        effect_size=proportion_effectsize(est_test_conv, est_control_conv),
        nobs1=None,
        alpha=alpha,
        power=power,
        alternative=alternative,
        ratio=1
    )

    return control_size


def power_graph(
    p_null:float,
    p_alt:float,
    n:int,
    alpha:float=0.05,
    plot:bool=True,
    null_color:str='#757471',
    alt_color:str='#22ad10',
    kpi:str='Conversion'):

    """
        Input:
            p_null: base success rate under null hypothesis
            p_alt : desired success rate to be detected, must be larger than
                    p_null
            n     : number of observations made in each group
            alpha : Type-I error rate (rejection region for null)
            plot  : boolean for whether or not a plot of distributions will be
                    created
            colors: either name like 'red', 'black' or hex like '#0a7hb3'
            kpi   : name of kpi you are testing usually conversion will
                go on the x axis of graph so label intuitively

        Output:
            power : Power to detect the desired difference, under the null.
            Example: power_graph(.02, .025, 10845)
    """

    # Compute the power

# for se calculations of two samples visit
# https://sphweb.bumc.bu.edu/otlt/MPH-Modules/BS/BS704_Power/BS704_Power_print.html
    se_null = np.sqrt( (p_null * (1-p_null) + p_alt * (1-p_alt)) / n)
    null_dist = stats.norm(loc = p_null, scale = se_null)
    if p_alt > p_null:
        p_crit = null_dist.ppf(1 - alpha)
    else:
        p_crit = null_dist.ppf(alpha)

    se_alt  = np.sqrt( (p_null * (1-p_null) + p_alt * (1-p_alt)) / n)
    alt_dist = stats.norm(loc = p_alt, scale = se_alt)
    if p_alt > p_null:
        beta = alt_dist.cdf(p_crit)
    else:
        beta = 1-alt_dist.cdf(p_crit)

    if plot:
        # Compute distribution heights
        if p_alt > p_null:
            low_bound = null_dist.ppf(.01)
            high_bound = alt_dist.ppf(.99)
        else:
            high_bound = null_dist.ppf(.99)
            low_bound = alt_dist.ppf(.01)
        x = np.linspace(low_bound, high_bound, 201)
        y_null = null_dist.pdf(x)
        y_alt = alt_dist.pdf(x)

        # Plot the distributions
        plt.style.use('ggplot')
        plt.figure(figsize=(9,6))
        plt.plot(x, y_null, color=null_color)
        plt.plot(x, y_alt, color=alt_color)
        plt.vlines(p_crit, 0, np.amax([null_dist.pdf(p_crit), alt_dist.pdf(p_crit)]),
                   linestyles = '--', linewidth=1.5, color=null_color)
        if p_alt > p_null:
            plt.fill_between(x, y_alt , 0, where = (x >= p_crit - p_crit*.001), alpha = .2, color=alt_color)
        else:
            plt.fill_between(x, y_alt , 0, where = (x <= p_crit + p_crit*.001), alpha = .2, color=alt_color)
        print()
        plt.title('Test Power', color='#686b6e', pad=30)
        plt.legend(['Null','Alternative'])
        plt.xlabel('\nKPI: {}\n\nTest Power: {}'
                   .format(kpi, round((1-beta),2)))
        plt.ylabel('Density')
        plt.show()

    # return power
    print('\nPower of test: {}'.format(round((1-beta),2)))

    return (1 - beta)


def bootstrap_point_estimator(
    data:list,
    n_samples:int=10_000,
    estimator:str='Mean',
    quant:float=None,
    return_all:bool=False,
    color:str='#22ad10',
    est_color:str='red') -> 'distribution, point est, opt: Conf Int.':

    '''
        Input:
            1. data: list-like or pd.Series(list)
            2. n_samples: number of bootstrap samples to draw With Repl.
            3. estimate: point estimate of interest 'Mean','Median','Quantile'
            4. quant: optional-Set to 'quantile' to estimate a quantile [0,1]
            5. return_all: bool - default False
                If False returns average, if true returns 3-tuple of
                estimate and 95% CI
            6. color: hex or 'red' as an example
            7. est_color: color of vertical line for estimate
        Output:
            1. Historgram and statistics of the bootstrapped
                point estimate of interest
            2. int or 3-tuple of points of interest
                (lower 95% int, average, upper 95% int)

        Example: bootstrap_point_estimator(my_list, 1_000, 'Mean')
    '''

    if type(data) == list:
        data = pd.Series(data)
#         make some if else about if the datatype is something other
#         than a list or pandas series to break and return statement

    est_list = list()
    for i in range(n_samples):
        sample_i = data.sample(
            frac=1,
            replace=True,
            random_state=i)

        if estimator == 'Mean':
            est_list.append(sample_i.mean())
            q=''
        elif estimator == 'Median':
            est_list.append(sample_i.median())
            q=''
        elif estimator == 'Quantile':
            est_list.append(sample_i.quantile(quant))
            q=' (' + str(quant) + ') '
        else:
            print('estimator argument must be "mean", "median"'
                  ' or "quantile"')
            q=''

    plt.figure(figsize=(8,5))
    plt.style.use('ggplot')
    _ = plt.hist(est_list, color=color, alpha=.3)
    plt.axvline(np.median(est_list), linestyle='--', color=est_color)
    plt.annotate(
        'Average='+str(round(np.median(est_list),2)),
        (np.median(est_list), n_samples/10),
        rotation=0)
    plt.axvline(np.quantile(est_list, .025), linestyle='--', color=color)
    plt.annotate(
        '',
        (np.quantile(est_list,.025)-np.quantile(est_list,.025)*.026, n_samples/5),
        rotation=0)
    plt.axvline(np.quantile(est_list, .975), linestyle='--', color=color)
    plt.annotate(
        '',
        (np.quantile(est_list, .975), n_samples/6),
        rotation=0)
    plt.title('\nBootstrapped Estimate of the {0}\n'
               '95% Conf. Int. Shown [{2} - {3}]\n'
               .format(
                   estimator + q,
                   n_samples,
                   round(np.quantile(est_list, .025),2),
                   round(np.quantile(est_list, .975),2)),
              color='#616161', fontsize=10)
    plt.ylabel('Density')
    plt.xlabel('Estimated ' + str(estimator + q) + ' Based on Sampling\n(' +
                str(n_samples) + ' Samples)')
    plt.show();

    if return_all == True:
        return(
            np.quantile(est_list, .025),
            np.median(est_list),
            np.quantile(est_list, .975))
    else:
        return(np.median(est_list))


def _create_bs_sample_dist(series_1, series_2, n_samples):
    '''
        Internal f(x) - Not to be called directly!
    '''

    series_1 = pd.Series(series_1)
    series_2 = pd.Series(series_2)

    tmp1 = list()
    tmp2 = list()
    for i in range(n_samples):
        tmp1.append(pd.Series(series_1).sample(replace=True, frac=1, random_state=i).mean())
        tmp2.append(pd.Series(series_2).sample(replace=True, frac=1, random_state=i).mean())
    return pd.Series(tmp1), pd.Series(tmp2)

def plot_bootstrap_dist_comparison(
        series_1:pd.Series,
        series_2:pd.Series,
        n_samples:int=100,
        x_axis_label:str='rename_x_axis_label',
        plot_title:str='rename_plot_title',
        control_group_name:str='Control',
        test_group_name:str='Test Group',
        control_color:str='#E97858',
        test_color:str='#1B6CA2') -> 'Test Distribution Plot':

    '''
        Input:
            series_1 - control series of test data
            series_2 - test  '' '' '' ''
            n_samples - the number of samples to be drawn
            plot config options:
                x_axis_label
                plot_title
                control_group_name (shows up in legend)
                test_group_name (legend)
                control_color
                test_color
        Output:
            Test and control distributions with mean and CI
        Example:
            plot_bootstrap_dist_comparison(list1, list2, n_samples=500)

    '''

    samp1, samp2 = _create_bs_sample_dist(series_1, series_2, n_samples=n_samples)

    plt.style.use('ggplot')
    plt.figure(figsize=(10,4))
    plt.hist(samp1, label=control_group_name, alpha=.4, color=control_color)
    plt.hist(samp2, label=test_group_name, alpha=.4, color=test_color)
    plt.ylabel('Frequency')
    plt.xlabel(x_axis_label)
    plt.title('\n' + plot_title + '\n', color='#596168')

    tmp_line_height = len(samp1)*.33
    plt.vlines(np.quantile(samp1,.05), 0, tmp_line_height, color=control_color, alpha=.6, linestyles='dotted')
    plt.vlines(samp1.mean(), 0, tmp_line_height, color=control_color, alpha=.6, linestyles='dashed')
    plt.vlines(np.quantile(samp1,.95), 0, tmp_line_height, color=control_color, alpha=.6, linestyles='dotted')
    plt.vlines(np.quantile(samp2,.05), 0, tmp_line_height, color=test_color, alpha=.6, linestyles='dotted')
    plt.vlines(samp2.mean(), 0, tmp_line_height, color=test_color, alpha=.6, linestyles='dashed')
    plt.vlines(np.quantile(samp2,.95), 0, tmp_line_height, color=test_color, alpha=.6, linestyles='dotted')

    plt.annotate('Mean\n{0:0.2f}'.format(samp1.mean()), (samp1.mean(), tmp_line_height*.8), color=control_color)
    plt.annotate('Mean\n{0:0.2f}'.format(samp2.mean()), (samp2.mean(), tmp_line_height*.9), color=test_color)


    plt.legend();
