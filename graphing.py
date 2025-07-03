# @Author agrgordon
# Last updated: July 3 2025

import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import patches
import matplotlib
import seaborn as sb
import numpy as np
from statannotations.Annotator import Annotator as annot
from scipy import stats
from scikit_posthocs import posthoc_tukey

defaultcolor = 'bright'
black = '#000000'

all_colors = matplotlib.colors.CSS4_COLORS
color_names = list(all_colors)

# Returns a barplot of y vs x with data points shown
# hue and palette are settable per normal seaborn usage
#    you can submit an incomplete palette and it will finished per the color scheme

# yscale is either 'linear' or 'log' and determines the estimator and scaling of the y axis
# order can be set to either "asc" or "desc" to plot the values in ascending or descending order respectively
#    order can also be a full array of the values to plot

# dodge should be set to True for grouped bar plots (otherwise False)

# function returns a tuple (bar, order) which is the mat plot lib axes and the x axis ordering

def niceplot(data,x,y,*,hue=None, yscale='linear',dodge=False, order=None, palette={}, 
             color_scheme=defaultcolor, labelangle = 45, dotsize = 5, capsize=0.3, nobars = False, 
             legend = True, debug = False, **kwargs):
    
    # defines the estimator plotted for each set of observations
    # default is arithmetic mean, geometric mean is used if graph is in logarithmic scale
    if yscale == 'log':
        estimator = lambda x: np.exp(np.mean(np.log(x)))
    else:
        estimator = 'mean'
    
    if order == 'asc':
        result = data.groupby([x])[y].aggregate(estimator).reset_index().sort_values(y)
        order = result[x]
        order = order.to_numpy()
    elif order == 'desc':
        result = data.groupby([x])[y].aggregate(estimator).reset_index().sort_values(y, ascending=False)
        order = result[x]
        order = order.to_numpy()
    elif order == None:
        result = data.groupby([x])[y].aggregate('min').reset_index()
        order = result[x]
        order = order.to_numpy()

    # this line seems unnecessary but allows for x values to be numbers
    if hue == None:
        hue = x
    
    palette = autopalette(data=data, x=x, y=y, hue=hue, order=order, partialpal=palette, color_scheme=color_scheme, debug = debug)
    
    dot = sb.swarmplot(data=data,x=x,y=y,hue=hue, palette=palette,
                   edgecolor='black', linewidth = 1, dodge=dodge, order=order,  size = dotsize, 
                   legend = False, **kwargs
                   )
    
    if nobars:
        bar = sb.barplot(data=data,x=x,y=y,estimator=estimator,
            hue=hue, palette=palette, order=order, #basic graph creation
             dodge=dodge, errorbar=None,
            linewidth = 3, edgecolor = black, #coloring
            legend = legend,
            **kwargs)
        
    else:
        bar = sb.barplot(data=data,x=x,y=y,estimator=estimator,
                hue=hue, palette=palette, order=order, #basic graph creation
                err_kws={'linewidth': 2},
                errorbar=('se',1),
                capsize = capsize, dodge=dodge,
                linewidth = 3, edgecolor = black, #coloring
                legend = legend,
                **kwargs
              )

    
    #The below lines fix the colors for bars and error bars
    
    patches = bar.patches
    
    if not nobars:
        # Error bars
        for i, line in enumerate(bar.get_lines()):
            newcolor = patches[i].get_facecolor()
            line.set_color(newcolor)

    # Shading
    for patch in patches:
        newcolor = patch.get_facecolor()
        patch.set_edgecolor(newcolor)
        newcolor = sb.set_hls_values(newcolor, l = .9)
        patch.set_facecolor(newcolor)
    
    sb.despine(ax=bar)
    
    ha = 'left'
    va = 'top'
    if labelangle == 0:
        ha = 'center'
    elif(labelangle > 0):
        ha = 'right'
    if labelangle == 90 or labelangle ==-90:
        va = 'center'

    curticks = bar.get_xticks();
    bar.set_xticks(curticks, order, rotation = labelangle,ha=ha, va=va, rotation_mode = 'anchor');
    bar.set(yscale=yscale)
    if legend == True:
        bar.legend()
    
    return (bar, order)

# Returns a dose response chart for data varying in a numeric quality (x) and categorical quality (hue)
# order determines how the colors are assigned to each variable
#     order must be a list of values or None
def doseResponse(data,x,y,*,hue=None, yscale='linear',dodge=False, order=None, palette={},
                 color_scheme=defaultcolor, nomarks = False, markers = None, dashes = None,
                 **kwargs):
    
    # defines the estimator plotted for each set of observations
    # default is arithmetic mean, geometric mean is used if graph is in logarithmic scale
    estimator = 'mean'
    if yscale == 'log':
        estimator = lambda x: np.exp(np.mean(np.log(x)))
    else:
        estimator = 'mean'
 
    palette = autopalette(data=data, x=x, y=y, hue=hue, order=order, partialpal=palette, color_scheme=color_scheme)
    
    # arbitrarily assign marker and dash options    
    markeroptions = ['o','v','^','s','p','P','*','h','X','D'] #selected from matplotlib markers
    dash = (3,1)
    
    if markers == None and not nomarks:
        markers = palette.copy()
        counter = 0
        for key in markers.keys():
            markers[key] = markeroptions[counter]
            counter = (counter + 1) % len(markeroptions)
            
    if dashes == None:
        dashes = palette.copy()
        for key in dashes.keys():
            dashes[key] = dash
        
    if not nomarks:
        dot = sb.scatterplot(data=data,x=x,y=y,hue=hue,
                 palette = palette, style = hue,
                 edgecolor='black', linewidth = 1, markers=markers,
                **kwargs
                      )

    line = sb.lineplot(data=data,x=x,y=y,hue=hue,
            palette = palette, estimator = estimator, errorbar=("se",1),
            style = hue, dashes = dashes,
            **kwargs)

    legendhands, labels = line.get_legend_handles_labels()

    count = int((len(labels)+1) / 2)

    if not nomarks:
        new_handles = [matplotlib.lines.Line2D((0,1),(0,1), 
                            color = legendhands[i+count].get_color(), lw = legendhands[i+count].get_linewidth(),
                            marker = legendhands[i].get_marker(), 
                            dashes = dashes[labels[i]],
                            markerfacecolor = legendhands[i].get_markerfacecolor(),
                            markersize = legendhands[i].get_markersize()) 
                            for i in range(count)]
        
        line.legend(handles=new_handles,labels=labels[:count])

    sb.despine()
    line.set(yscale=yscale)
    
    return line


# Accepts a partial color palette and returns an appropriate full palette
def autopalette(data, x, *, y='none', hue=None, order='default', partialpal={},
                color_scheme='bright', debug = False):
    
    if isinstance(hue, list):
        keys = hue
    else:
        if hue == None:
            hue = x

        if isinstance(order, str):
            if order == 'default':
                keys = data[hue]
            else:
                raise Exception("Invalid order option \"" + order + '"' )
        else:
            keys = order

        if hue != x:
             keys = list(data[hue])

    if debug:
        print("The used keys are: " + str(keys))
        print("Recieved partial palette" + str(partialpal))

    pal = {}
    for k in partialpal.keys():
        if k in keys:
            pal[k] = partialpal[k]

    if debug:
        print("After partial pal validation, the palette is: " + str(pal))
    
    if color_scheme in color_names:
        colors = [all_colors[color_scheme]] * (len(keys) - len(pal))
    else:
        colors = sb.color_palette(color_scheme, len(keys) - len(pal))
    
    col = 0
     
    for name in keys:
        if name not in pal:
            pal[name] = colors[col]
            col += 1
            if debug:
                print(str(name) + " has been successfully assigned to " + str(matplotlib.colors.to_hex(colors[col - 1])))

    if debug:
        print("Final color palette: " + str(pal))
            
    return pal

# Plots a barchart with points (using niceplot) using data equal to the y value divided by the sum of the reference columns
# Useful for e.g. creating proportional flow data from +/- data
def plotrelative(data, x, y, refcols, hue=None, yscale='linear',dodge=False, order=None, palette={}, color_scheme=defaultcolor):
    calcvalues = data.copy()
    
    new_y = "Fraction " + y
    
    calcvalues[new_y] = calcvalues[y].divide(calcvalues[refcols].sum(axis=1), axis=0)
    
    (plot, order) = niceplot(data=calcvalues, x=x, y=(new_y), hue=hue, 
                    yscale=yscale, dodge=dodge, order=order, 
                    palette=palette, color_scheme=color_scheme)
    
    return (plot, order, calcvalues)

# Returns a dataframe with all rows removed except those containing a value matching keys in the indicated column
# Useful for simplifying huge messy graphs
# Now just wrapper functions for the correct pandas calls
def dropallexcept(data, column, keys):

    return data[data[column].isin(keys)].reset_index(drop = True)

def droponly(data, column, keys):

    return data[~data[column].isin(keys)].reset_index(drop = True)

'''
draws a line of best fit for y vs x, and displays the equation. 
Mostly good because faster than excel. 
Does not currently handle multiple datasets/hue/row/col variables well
Will hopefully implement at least minimal functionality in the future
'''
def linreg(data, x, y, *, color = 'black', yscale = 'linear', equation = False, 
               eqxpos = 0.5, eqypos = 0.1, **kwargs):
    
    #These lines generate the plot itself and do some minor formating adjustments
    lm = sb.regplot(data = data, x = x, y = y, color = color, **kwargs)
    
    lm.set(yscale = yscale)
    sb.despine()
    
    #These lines display the equation for the line of best fit
    if(equation != False):
        fit = np.polyfit(x=data.loc[:, x], y=data.loc[:, y], deg=1)

        eq = 'y = {0:.3E} x + {1:.3E}'.format(fit[0],fit[1])
        r = np.corrcoef(x = data.loc[:,x], y=data.loc[:, y])
        r2 = r[0,1] * r[1,0]

        rsquared = ('r$^2$ = {0:.4f}'.format(r2))
        if equation == 'just r':
            info = rsquared
        else:
            info = eq + "\n" + rsquared

        plt.text(eqxpos, eqypos, info, ha = 'center', va = 'center', 
                 transform = lm.transAxes)
    
    return lm


# scatter plot with error bars and some minor visual adjustments

# autopalette line currently bugged, ostensibly everything else works fine
def scatter(data, x, y, *, color_scheme = defaultcolor, yscale = 'linear', 
            hue=None, palette={},
            xerr=0, yerr=0, ecolor=black, elinewidth=0.5, capsize=3, s=100,
            **kwargs):
    
    #currently does not allow for ordering after data is passed -- work on this?
    palette = autopalette(data=data, x=x, y=y, hue=hue, partialpal=palette,
                          color_scheme=color_scheme)
    
    markeroptions = ['o','v','^','s','p','P','*','h','X','D'] #selected from matplotlib markers
    
    if color_scheme == 'Paired' :
        markeroptions = ['o', 'o', '^', '^', 's', 's', 'P', 'P', '*', '*',
                         'h', 'h', 'X', 'X', 'D', 'D']
    
    markers = palette.copy()
    
    counter = 0
    for key in markers.keys():
        markers[key] = markeroptions[counter]
        counter += 1

    graph = sb.scatterplot(data=data, x=x, y=y, hue=hue, palette=palette,
                           s=s, style=hue, markers = markers)
    
    # note, if 'color' is not set to 'None,' then it connects the points and looks horrible
    if(xerr > 0):
        graph.errorbar(x=data[x], y=data[y], color = 'None',
            xerr=xerr, ecolor=ecolor, elinewidth=elinewidth, capsize = capsize)
        
    if(yerr > 0):
        graph.errorbar(x=data[x], y=data[y], color = 'None',
            xerr=xerr, ecolor=ecolor, elinewidth=elinewidth, capsize = capsize)
    
    graph.set(yscale = yscale)
    sb.move_legend(graph, "upper left", bbox_to_anchor=[1,1])
    sb.despine()
    
    return graph

# this function currently does not plot error
# takes a dataframe (data), x column (x), y column (y), and grouping column (hue)
# plots x vs y where each point is the x- and y-average of all rows in data with a given hue
# plots the line of best fit through the scatterplot with optional display of equation and r^2
# optionally cluster data relative to controls for sets of observations based on normcolumn and normhue
def scatteravg(data, x, y, hue, *, 
               color_scheme = defaultcolor, yscale = 'linear', equation = False, 
               normcolumn = '', normhue = '', **kwargs):
        
    # identify normalization averages
    if normhue != '' and normcolumn != '':
        
        yavgs = normalizecol(data, y, hue, normcolumn, normhue)
        xavgs = normalizecol(data, x, hue, normcolumn, normhue)
        
        xavg = []
        yavg = []
        
        for key in xavgs.keys():
            if key in yavgs.keys():
                xavg += [xavgs[key]]
                yavg += [yavgs[key]]
                
    # or just identify averages
    else:
        huexcounts = {}
        hueycounts = {}
        xsums = {}
        ysums = {}
        for i in range(len(data[hue])):

            if data[hue][i] in huexcounts.keys():
                if str(data[x][i]).lower() != 'nan' :
                    xsums[data[hue][i]] += float(data[x][i])
                    huexcounts[data[hue][i]] += 1

            else:
                if str(data[x][i]).lower() != 'nan' :
                    xsums[data[hue][i]] = float(data[x][i])
                    huexcounts[data[hue][i]] = 1

            if data[hue][i] in hueycounts.keys():
                if str(data[y][i]).lower() != 'nan' :
                    ysums[data[hue][i]] += float(data[y][i])
                    hueycounts[data[hue][i]] += 1

            else:
                if str(data[y][i]).lower() != 'nan' :
                    ysums[data[hue][i]] = float(data[y][i])
                    hueycounts[data[hue][i]] = 1
    
        # find the average of each variable and make an averages dataframe

        xavg = []
        yavg = []

        for key in huexcounts.keys():
            if key in hueycounts.keys():
                xavg += [xsums[key] / huexcounts[key]]
                yavg += [ysums[key] / hueycounts[key]]
        
    newdf = pd.DataFrame(list(zip(xavg, yavg)), columns = [x, y])
    
    # print(newdf)
    
    # plot the set of average points using linreg above
    
    graph = linreg(data = newdf, x=x, y=y, 
                   color_scheme = color_scheme, yscale = yscale, equation = equation, **kwargs)
    
    # add the 2D error bars to the final graph
    
    
    return graph

# Takes as inputa dataframe, a list of x columns and a list of y columns
# Returns a dataframe with y values averaged across replicates defined by x values
# y columns must be numeric
# any columns not listed in xs or ys will be lost

def averagedf(data, xs, ys, debug = False):
    xvals = []
    sums = []
    ycts = []

    for i in data.index:
        new_key = [data.loc[i,x] for x in xs]

        if debug:
            print(new_key)
        
        if not new_key in xvals:
            xvals += [new_key]
            sums += [[0 for y in ys]]
            ycts += [[0 for y in ys]]

        index = xvals.index(new_key)

        for j in range(len(ys)):
            try:
                sums[index][j] += float(data.loc[i, ys[j]])
                ycts[index][j] += 1
            except:
                pass
    
    for k in range(len(xvals)):
        for y in range(len(ys)):
            if ycts[k][y] == 0:
                print("The count for x values " + str(k) + " was zero")
            else:
                sums[k][y] = sums[k][y]/ycts[k][y]

    dfdict = {}

    for i in range(len(xs)):
        dfdict[xs[i]] = []
        for j in range(len(xvals)):
            dfdict[xs[i]] += [xvals[j][i]]

    for i in range(len(ys)):
        dfdict[ys[i]] = []
        for j in range(len(xvals)):
            dfdict[ys[i]] += [sums[j][i]]

    return pd.DataFrame(dfdict)



# takes a dataframe, y column title, hue column title, grouping column title, and normalizing hue
# returns a list of the average normalized values by hue
# this function complies with the slices of dataframe issues and should not raise such warnings
def normalizecol(data, y, hue, normcolumn, normhue, debug = False):
    
    # identify the normalization values
    ynormsums = {}
    normycounts = {}

    for i in range(len(data[hue])):
        if data.loc[i, hue] == normhue and str(data.loc[i, y]).lower() != 'nan':

            if data.loc[i, normcolumn] in normycounts.keys():
                ynormsums[data.loc[i, normcolumn]] += float(data.loc[i, y])
                normycounts[data.loc[i, normcolumn]] += 1

            else:
                ynormsums[data.loc[i, normcolumn]] = float(data.loc[i, y])
                normycounts[data.loc[i, normcolumn]] = 1
                    
    if debug:
        print(ynormsums)
        print(normycounts)
    
    ynorms = {}

    for key in normycounts.keys():
        ynorms[key] = ynormsums[key] / normycounts[key]

    if debug:
        print(ynorms)
    
    # find the averages
    hueycounts = {}
    ysums = {}
    
    for i in range(len(data[hue])):

        if str(data.loc[i, y]).lower() != 'nan':
            ynorm = ynorms[data.loc[i, normcolumn]]
        
            if data.loc[i, hue] in hueycounts.keys():
                ysums[data.loc[i, hue]] += float(data.loc[i, y]) / ynorm
                hueycounts[data.loc[i, hue]] += 1
                
            else:
                ysums[data.loc[i, hue]] = float(data.loc[i, y]) / ynorm
                hueycounts[data.loc[i, hue]] = 1

    if debug:
        print(ysums)
        print(hueycounts)
    
    for key in hueycounts.keys():
        ysums[key] /= hueycounts[key]
        
    return ysums

# returns data frame with columns normalized based on the indicated 
# normalization hue
# data = the dataframe to be normalized
# ys = a single column header or list of column headers for the column(s) to be normalized
# hue = the column header for the grouping column (often x values)
# normhue = the value in the 'hue' column to use as the normalization metric
# normcolumn = an optional subgrouping to determine normalizations (e.g. date of experiment)
# this function complies with the slices of dataframe issues and should not raise such warnings
def normalizedf(data, ys, hue, normhue, normcolumn = None, log = False, debug = False):

    # sanitize data
    if not isinstance(ys, list):
        ys = [ys]
    
    # identify the normalization values
    ynormsums = {}
    normycounts = {}
    ynorms = {}

    for y in ys:
        ynormsums[y] = {}
        normycounts[y] = {}
    
        for i in range(len(data[hue])):
            
            if normcolumn == None:
                ref = 'none'
            else:
                ref = data.loc[i, normcolumn]
            
            if data.loc[i, hue] == normhue and str(data.loc[i, y]).lower() != 'nan':
    
                if ref in normycounts[y].keys():
                    if log:
                        ynormsums[y][ref] *= float(data.loc[i,y])
                    else:
                        ynormsums[y][ref] += float(data.loc[i,y])
                    normycounts[y][ref] += 1
    
                else:
                    ynormsums[y][ref] = float(data.loc[i, y])
                    normycounts[y][ref] = 1
                    
        ynorms[y] = {}

        if log:
            for key in normycounts[y].keys():
                ynorms[y][key] = np.power(ynormsums[y][key], 1 / normycounts[y][key])
        else:
            for key in normycounts[y].keys():
                ynorms[y][key] = ynormsums[y][key] / normycounts[y][key]
        
    # once we have the normalizations, modify a copy of the dataframe

    if debug:
        print('ynorms = ' + str(ynorms))
    
    newdf = data.copy()
    
    for i in range(len(newdf[hue])):
        if normcolumn == None:
            ref = 'none'
        else:
            ref = data.loc[i, normcolumn]
        
        for y in ys:
            if str(newdf.loc[i, y]).lower() != 'nan':
                newdf.loc[i, y] = newdf.loc[i, y] / ynorms[y][ref]
        
    return newdf
        
    

# takes a dataframe and performs a one-way ANOVA statistical comparison with tukey's post-hoc correction
# can optionally concatenate data from two different columns (e.g. excipient type and concentration)
# can optionally return only the p values matching a requested set of pairs for integration with statannot
def anova(df, y, columnA, *, columnB = None, pairs = None):
    # do data prep for anova + tukey
    dupe_data = df.copy()
    
    if columnB != None:
        dupe_data = df.rename(columns={columnA : 'A', columnB : 'B'})
    else:
        dupe_data = df.rename(columns={columnA : 'A'})
    
    # concatenate the two variables of interest into one column    
    concatVals = []
    concatKeys = []
    
    for i in range(len(dupe_data['A'])):
        if columnB != None:
            concatKeys += [str(dupe_data['A'][i]) + " " + str(dupe_data['B'][i])]
        else:
            concatKeys += [dupe_data['A'][i]]
        concatVals += [dupe_data[y][i]]
    
    newdf = pd.DataFrame(list(zip(concatKeys, concatVals)), columns = ['key', y])
    
    # do the stats
    keys = []
    for k in concatKeys:
        if k not in keys:
            keys += [k]

    data = []

    for k in keys:
        data.append(newdf[newdf.key == k][y])
    
    stats.f_oneway(*data)
    
    tukey_df = posthoc_tukey(newdf, val_col=y, group_col="key")
    
    # clean up the tukey array
    remove = np.tril(np.ones(tukey_df.shape), k=0).astype("bool")
    tukey_df[remove] = np.nan

    molten_df = tukey_df.melt(ignore_index=False).reset_index().dropna()
    
    # get the relevant data
    tukeypairs = [(i[1]["index"], i[1]["variable"]) for i in molten_df.iterrows()]
    tukeyp_values = [i[1]["value"] for i in molten_df.iterrows()]
    
    if pairs == None:
        return (tukeypairs, tukeyp_values)
    
    p_values = []
    
    for pair in pairs:
        if columnB != None:
            groupa = str(pair[0][0]) + " " + str(pair[0][1])
            groupb = str(pair[1][0]) + " " + str(pair[1][1])
        else:
            groupa = pair[0]
            groupb = pair[1]
        
        if (groupa, groupb) in tukeypairs:
            p_values += [tukeyp_values[tukeypairs.index((groupa,groupb))]]
        elif (groupb, groupa) in tukeypairs:
            p_values += [tukeyp_values[tukeypairs.index((groupb,groupa))]]
        else:
            p_values += [1]
            print('Stats error!')
            
    for i in range(len(p_values)):
        if p_values[i] == 1e-3: # lowest value this test will give
            p_values[i] = 0.99999999e-3 # necessary for correct presentation by statannot
    
    return p_values

# annotates p values on input axis
# axis = the matplotlib axis object to annotate
# data = the dataframe corresponding to the graph, used to determine ymax
# y = the y column, used to determine ymax
# pvals = the p values as a list of floats
# xvals = the x values as a list of either tuple pairs or single values
# inv = if the data is all negative
# vals = display p values instead of stars
# lines = whether to display lines for non-spanning p values
# showNS = whether to label non-significant values
# TODO cleaner spacing for multiple comparisons, making the inverse mode work correctly
# TODO auto-resize graph axes if comparisons will be drawn outside the axis
def pannot(axis, data, y, pvals, xvals, inv = False, vals = False, lines = True, showNS = True, debug = False):
    # find the maximum (or minimum) y value so we annotate above it
    ymax = data.loc[0, y]

    s = 1
    if inv:
        s = -1

    for i in range(len(data[y])):
        if s * data.loc[i,y] > s * ymax or str(ymax) == 'nan':
            ymax = data.loc[i,y]

    if debug:
        print("Found maximum y-value of " + str(ymax))

    xpos = {}

    barwidth = 0

    # identify unique xvalues
    for x in xvals:
        if debug:
            print("Finding axis coordinates for x value(s): " + str(x) + "...")
        
        if isinstance(x, tuple):
            if barwidth == 0 and ( isinstance(x[0], tuple) or isinstance(x[1], tuple) ):
                barwidth = axis.patches[0].get_width() / (axis.get_xlim()[1] - axis.get_xlim()[0])

                if debug:
                    print('determined barwidth to be ' + str(barwidth))
                
            xpos[x[0]] = findAxPos(axis, x[0], ymax, barwidth)[0]
            xpos[x[1]] = findAxPos(axis, x[1], ymax, barwidth)[0]
        else:
            xpos[x] = findAxPos(axis, x, ymax)[0]

        if debug:
            print("added axis coordinate value of " + str(xpos[x]))

    # order all x values by position on the x axis
    xref = []
    xdists = []
    for x in xpos.keys():
        xref += [x]
        xdists += [xpos[x]]

    xdists, xrefs = zip(*sorted(zip(xdists, xref)))

    xrefs = list(xrefs)
    xdists = list(xdists)

    # identify length of connecting bars
    pdists = []

    for i in range(len(pvals)):
        if isinstance(xvals[i], tuple):
            if debug:
                print(xvals[i])
            pdists += [abs(xpos[xvals[i][0]] -  xpos[xvals[i][1]])]
        else:
            pdists += [0]

    # sort p values from shortest to longest distance of bar
    # built-in function seems to be buggy with mixed-type xvals(?)
    sortlist = [[pdists[0], pvals[0], xvals[0]]]

    for i in range(1,len(pdists)):
        flag = False
        for j in range(len(sortlist)):
            if pdists[i] < sortlist[j][0]:
                sortlist = sortlist[:j] + [[pdists[i], pvals[i], xvals[i]]] + sortlist[j:]
                flag = True
                break
        if not flag:
            sortlist += [[pdists[i], pvals[i], xvals[i]]]

    xvals = [tup[2] for tup in sortlist]
    pvals = [tup[1] for tup in sortlist]

    '''
    pdists, pvals, xvals = zip(*sorted(zip(pdists, pvals, xvals)))

    xvals = list(xvals)
    pvals = list(pvals)
    '''

    # re-initialize 
    for x in xpos.keys():
        xpos[x] = []
    
    if debug:
        print('ymax = ' + str(ymax))

    if vals:
        fontsize = 'small'
        for i in range(len(pvals)):
            if pvals[i] == 0.9:
                pvals[i] = '> 0.9'
            elif pvals[i] <= 0.001:
                pvals[i] = '< 0.001'
            else:
                pvals[i] = "{:.3f}".format(pvals[i])
    else:
        fontsize = 'medium'
        for i in range(len(pvals)):
            if pvals[i] > 0.05:
                pvals[i] = 'ns'
                if not showNS:
                    pvals[i] = ''
            elif pvals[i] > 0.01:
                pvals[i] = '*'
            elif pvals[i] > 0.001:
                pvals[i] = '**'
            elif pvals[i] > 0.0001:
                pvals[i] = '***'
            else:
                pvals[i] = '****'

    xlabels = axis.get_xticklabels()

    incr = s * 0.08

    for i in range(len(pvals)):
        offset = s*0.05
        if '*' in pvals[i]:
            offset = s*0.07 - 0.035
        if isinstance(xvals[i], tuple):
            
            # finds the lowest layer not used by any of the intervening x positions
            usedlayers = []
            foundcount = 0
            for x in xrefs:
                if x in xvals[i]:
                    foundcount += 1
                if foundcount >= 1:
                    usedlayers += xpos[x]
                if foundcount == 2:
                    break

            layer = 0
            while True:
                if layer not in usedlayers:
                    break
                layer += 1

            foundcount = 0
            for x in xrefs:
                if x in xvals[i]:
                    foundcount += 1
                if foundcount >= 1:
                    xpos[x] += [layer]
                if foundcount == 2:
                    break

            x1 = findAxPos(axis, xvals[i][0], ymax, barwidth)
            x2 = findAxPos(axis, xvals[i][1], ymax, barwidth)
            
            xy = ((x1[0] + x2[0]) / 2, x1[1] + offset + incr * layer)
            axis.add_patch(patches.Rectangle(xy = (x1[0], x1[1] + s*0.03 + (incr * layer)),
                                  height = 0.005, width = x2[0]-x1[0], 
                                  color='k', transform = axis.transAxes))
        else:
            layer = 0
            while True:
                if layer not in xpos[xvals[i]]:
                    break
                layer += 1
            xpos[xvals[i]] += [layer]
            xy = findAxPos(axis, xvals[i], ymax)
                
            if lines and pvals[i] != '':
                axis.add_patch(patches.Rectangle(xy=(xy[0] -0.05, xy[1] + s*0.03 + (incr * layer)), 
                                    height = 0.005, width = 0.1,
                                    color='k', transform = axis.transAxes))
            xy = xy + (0, offset + incr * layer)
        if debug:
            print('Annotating a p-value at ' + str(xy))
        axis.annotate(text = pvals[i], xy = xy, ha='center', xycoords = 'axes fraction', fontsize = fontsize)

# helper function that finds a position on a not-necessarily numerical x axis
# Note that this currently gets really buggy with seaborn logarithmic axes
# barwidth to be given in axes coordinates (i.e. should be between 0.01 and 0.5, roughly)
def findAxPos(axis, x, y, barwidth = 0):
    flag = False

    # parses x tuple coordinates for clustered bar graphs
    if isinstance(x, tuple):
        resp = findAxPos(axis, x[0], y)
        
        if barwidth == 0:
            return resp

        legendlabels = axis.get_legend_handles_labels()[1]

        offset = (-1) * barwidth * (len(legendlabels) - 1) / 2

        for l in legendlabels:
            if l == x[1]:
                break
            offset += barwidth

        return (resp[0] + offset, resp[1])
    

    # find x value based on axis labels 
    xlabels = axis.get_xticklabels()
    for l in xlabels:
        if l.get_text() == str(x):
            resp = axis.transLimits.transform((l.get_position()[0], y))
            flag = True
            break

    # find x value based on raw numbers
    if not flag:
        resp = axis.transLimits.transform((x,y))

    # if values seem crazy, assume it's a logarithmic axis
    if resp[1] > 1 or resp[1] < 0:
        return findAxPos(axis, x, np.log10(y))

    if resp[0] > 1 or resp[0] < 0:
        return findAxPos(axis, np.log10(x), y)
    
    return resp

# helper function that draws a statistical comparison hat
# from x1 to x2 at height y
