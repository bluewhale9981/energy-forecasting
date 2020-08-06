# -*- coding: utf-8 -*-
from __future__ import absolute_import

"""Collection of visualization utilities."""

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


# Methods
# -------

def _generate_data_types_df(data):
    """Help to generate a new dataframe includes each column as a data type of the input dataframe."""
    data_types = data.dtypes.to_frame().reset_index().rename(columns={'index': 'column_name', 0: 'data_type'})
    data_types['data_type'].apply(str)

    pivot_data_types = data_types.pivot(columns='data_type')
    pivot_data_types.columns = ['_'.join([x[0], str(x[1])]) for x in pivot_data_types.columns.values]

    temp = pd.concat([pivot_data_types[col].sort_values().reset_index(drop=True) for col in pivot_data_types],
                     axis=1, ignore_index=True)
    temp = temp.rename(columns={y[0]: y[1] for y in zip(temp.columns, pivot_data_types.columns)})
    temp = temp.dropna(how='all')

    return temp


def plot_stats(df, feature, target='TARGET', max_length_labels=3, label_rotation=False, horizontal_layout=True):
    """
    Generate a plot of the statistics between single feature with the target.

    Arguments:
        df (pd.DataFrame): The dataframe input.
        feature (str): Single feature field.
        target (str): The target field. Default is `TARGET`.
        max_length_labels: The maximum lenght of labels for decision roration label.
                           Default is 3.
        label_rotation (boolean): Default is `False`.
        horizontal_layout (boolean): Default is `True`.
    """
    temp = df[feature].value_counts()

    data = pd.DataFrame({feature: temp.index, 'Values': temp.values})

    # Calculate the percentage of target=1 per category value
    cat_perc = df[[feature, target]].groupby([feature], as_index=False).mean()
    cat_perc.sort_values(by=target, ascending=False, inplace=True)

    # Mark roration for label if the lenght of labels are more than max_length_labels
    if len(data[feature]) > max_length_labels:
        label_rotation = True

    # Mark horizontal for layout if the lenght of labels so really long
    if len(data[feature]) > (10*max_length_labels):
        horizontal_layout = False

    if horizontal_layout:
        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 6))
    else:
        fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(12, 14))

    sns.set_color_codes('pastel')
    s = sns.barplot(ax=ax1, x=feature, y='Values', data=data)

    if label_rotation:
        s.set_xticklabels(s.get_xticklabels(), rotation=90)

    s = sns.barplot(ax=ax2, x=feature, y=target, order=cat_perc[feature], data=cat_perc)

    if label_rotation:
        s.set_xticklabels(s.get_xticklabels(), rotation=90)

    plt.ylabel('Percent of target with value 1 [%]')
    plt.tick_params(axis='both', which='major')

    plt.show()


def plot_categorical(data, target='TARGET'):
    """
    Plot a single category field with the target. It includes three charts in a same frame.

    Arguments:
        data (dataframe): The dataframe need to visualize.
        target (str): The target string. Default is `TARGET`.
    """
    temp = _generate_data_types_df(data)

    try:
        object_cols = temp.column_name_object.loc[temp.column_name_object.notnull()]
    except AttributeError:
        object_cols = []

    for i in object_cols:
        cat_counts = data[i].fillna('Missing').value_counts()
        fig, ax = plt.subplots(1, 3, figsize=(12, 7))
        patches, text, _ = ax[0].pie(cat_counts, autopct='%.2f')
        ax[0].legend(patches, labels=cat_counts.index, loc='best')
        ax[0].set_title('% of Category in Whole Population')
        temp2 = data[[i, target]].fillna('Missing').groupby([i, target]).size().unstack(i).fillna(0)[cat_counts.index]
        temp2.plot(kind='bar', stacked=True, ax=ax[1])
        ax[1].set_title('# of Category in Target Population')
        temp2_per = temp2.div(temp2.sum(axis=1), axis=0)
        temp2_per.plot(kind='bar', stacked=True, ax=ax[2])
        ax[2].set_title('% of Category in Target Population')
        ax[2].legend_.remove()

        for t in temp2_per.index:
            cumsum = 0
            for c in temp2_per.columns:
                if temp2_per.loc[t, c] > 0.02:
                    ax[2].text(x=t, y=cumsum+temp2_per.loc[t, c]/2, s=round(temp2_per.loc[t, c]*100, 2))
                    cumsum = cumsum+temp2_per.loc[t, c]

        ax[0].legend_.remove()
        plt.suptitle(str(i))


def plot_dist_numerical(data, target='TARGET'):
    """
    Plot a distribution of a single numerical field with the target.

    Arguments:
        data (dataframe): The dataframe need to visualize.
        target (str): The target string. Default is `TARGET`.
    """
    temp = _generate_data_types_df(data)
    for col in temp.column_name_float64.dropna():
        try:
            plt.subplots()
            ax1 = plt.subplot2grid((3, 3), (0, 0), colspan=2)
            ax2 = plt.subplot2grid((3, 3), (1, 0), colspan=2)
            ax3 = plt.subplot2grid((3, 3), (2, 0), colspan=2)
            ax4 = plt.subplot2grid((3, 3), (0, 2), rowspan=3)
            sns.distplot(data[col].dropna(), ax=ax1)
            ax1.set_title('Prob. Dist. Full Population')
            sns.distplot(data.loc[data[target] == 0, col].dropna(), ax=ax2, color='green')
            ax2.set_title('Prob. Dist. %s=0 Population' % target)

            sns.distplot(data.loc[data[target] == 1, col].dropna(), ax=ax3, color='red')
            ax3.set_title('Prob. Dist. %s=1 Population' % target)

            data[[target, col]].boxplot(by=target, ax=ax4)
            fig = plt.gcf()
            fig.set_size_inches(16, 6)
            plt.subplots_adjust(hspace=.7)
            plt.subplots_adjust(wspace=.5)
            ax4.set_title('Box Plot')
            plt.suptitle(col)
        except Exception:
            pass


def kde_target(var_name, df, target='TARGET'):
    """ Plots the distribution of a variable colored by value of the target """

    # Calculate the correlation coefficient between the new variable and the target
    corr = df[target].corr(df[var_name])

    # Calculate medians for repaid vs not repaid
    avg_repaid = df.ix[df[target] == 0, var_name].median()
    avg_not_repaid = df.ix[df[target] == 1, var_name].median()

    plt.figure(figsize=(12, 6))

    # Plot the distribution for target == 0 and target == 1
    sns.kdeplot(df.ix[df[target] == 0, var_name], label='{} == 0'.format(target))
    sns.kdeplot(df.ix[df[target] == 1, var_name], label='{} == 1'.format(target))

    # label the plot
    plt.xlabel(var_name)
    plt.ylabel('Density')
    plt.title('%s Distribution' % var_name)
    plt.legend()

    # print out the correlation
    print('The correlation between %s and the %s is %0.4f' % (var_name, target, corr))
    # Print out average values
    print('Median value for loan that was not repaid = %0.4f' % avg_not_repaid)
    print('Median value for loan that was repaid =     %0.4f' % avg_repaid)
