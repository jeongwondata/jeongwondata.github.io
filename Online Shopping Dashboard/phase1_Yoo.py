#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from prettytable import PrettyTable
import statsmodels.api as sm
from mpl_toolkits.mplot3d import Axes3D
import warnings
warnings.filterwarnings('ignore')

#%%
url = 'online_shopping.csv'
df = pd.read_csv(url)

print(df.tail())
print(df.info())
#%% missing values
print(df.isnull().sum())

col31 = ['CustomerID','Gender','Location','Tenure_Months', 'Transaction_ID','Quantity','Avg_Price','Delivery_Charges',
        'Coupon_Status','GST','Offline_Spend','Online_Spend']
df.dropna(subset=col31, inplace=True)

print(df.isnull().sum())

#%% incoding and datetime

df['Transaction_Date'] = pd.to_datetime(df['Transaction_Date'])
df['CustomerID'] = df['CustomerID'].astype(int)
df['Transaction_ID'] = df['Transaction_ID'].astype(int)

#%%
df['Total_Spend'] = df['Offline_Spend'] + df['Online_Spend']
df['Revenue'] = df['Quantity'] * df['Avg_Price']

cat_cols = ['Gender', 'Location', 'Product_Category', 'Coupon_Status']
num_cols = ['Tenure_Months', 'Quantity', 'Avg_Price', 'Delivery_Charges',
            'GST', 'Offline_Spend', 'Online_Spend', 'Discount_pct','Total_Spend', 'Revenue']

#%%
df = df.drop(columns=['Unnamed: 0'])

print(f'{df.head()}\n{df.info()}\n{df.describe()}')

#%%
t4 = PrettyTable()
t4.title = 'Pearson Correlation'
sel = ['Avg_Price','Online_Spend','Offline_Spend','Delivery_Charges','Revenue']
corr_sel = df[sel].corr().round(2)
t4.field_names = [''] + sel
for idx in corr_sel.index:
    t4.add_row([idx] + [f"{v:.2f}" for v in corr_sel.loc[idx]])
print(t4)
#%%
PALETTE = 'Set2'
TITLE_COLOR = '#2c3e50'
LABEL_COLOR = '#34495e'
C_F = '#e74c3c'
C_M = '#3498db'
# ============================================================
# PLOT 1: Bar Plot - Sorted (Product Category)
# ============================================================
print('\n--- Plot 1: Sorted Bar Plot (Product Category Count) ---')
fig, ax = plt.subplots(figsize=(14, 7))
vc = df['Product_Category'].value_counts().sort_values(ascending=True)
colors = sns.color_palette(PALETTE, len(vc))
bars = ax.barh(vc.index, vc.values, color=colors)
for bar, val in zip(bars, vc.values):
        ax.text(val + 50, bar.get_y() + bar.get_height() / 2,
                f'{val:,}', va='center', fontsize=9, color=LABEL_COLOR)
ax.set_xlabel('Transaction Count', color=LABEL_COLOR)
ax.set_ylabel('Product Category', color=LABEL_COLOR)
ax.set_title('Product Category by Transaction Count (Sorted)', color=TITLE_COLOR, fontweight='bold')
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.show()

# ============================================================
# PLOT 2: Stacked Bar Plot (Gender × Location)
# ============================================================
print('\n--- Plot 2: Stacked Bar Plot (Gender × Location) ---')
fig, ax = plt.subplots(figsize=(10, 6))
ct = pd.crosstab(df['Location'], df['Gender'])
ct = ct.loc[ct.sum(axis=1).sort_values(ascending=False).index]
ct.plot(kind='bar', stacked=True, ax=ax, color=[C_F, C_M], edgecolor='white')
for c in ax.containers:
        ax.bar_label(c, fmt='%d', label_type='center', fontsize=9, color='white', fontweight='bold')
ax.set_title('Transactions by Location & Gender (Stacked)', color=TITLE_COLOR, fontweight='bold')
ax.set_xlabel('Location', color=LABEL_COLOR)
ax.set_ylabel('Count', color=LABEL_COLOR)
ax.legend(title='Gender');
ax.grid(axis='y', alpha=0.3)
plt.xticks(rotation=30);
plt.tight_layout()
plt.show()
# ============================================================
# PLOT 3: Grouped Bar Plot (Coupon Status × Gender)
# ============================================================
print('\n--- Plot 3: Grouped Bar Plot (Coupon Status × Gender) ---')
fig, ax = plt.subplots(figsize=(10, 6))
ct2 = pd.crosstab(df['Coupon_Status'], df['Gender'])
ct2.plot(kind='bar', ax=ax, color=[C_F, C_M], edgecolor='white')
for c in ax.containers:
        ax.bar_label(c, fmt='%d', fontsize=9, padding=2)
ax.set_title('Coupon Status by Gender (Grouped)', color=TITLE_COLOR, fontweight='bold')
ax.set_xlabel('Coupon Status', color=LABEL_COLOR)
ax.set_ylabel('Count', color=LABEL_COLOR)
ax.legend(title='Gender');
ax.grid(axis='y', alpha=0.3)
plt.xticks(rotation=0);
plt.tight_layout()
plt.show()
# ============================================================
# PLOT 4: Count Plot (Location with Gender hue)
# ============================================================
print('\n--- Plot 4: Count Plot (Location) ---')
fig, ax = plt.subplots(figsize=(10, 6))
order = df['Location'].value_counts().index
sns.countplot(data=df, x='Location', hue='Gender', order=order,
              palette=[C_F, C_M], ax=ax)
for c in ax.containers:
        ax.bar_label(c, fmt='%d', fontsize=9, padding=2)
ax.set_title('Count Plot: Transactions by Location & Gender', color=TITLE_COLOR, fontweight='bold')
ax.set_xlabel('Location', color=LABEL_COLOR)
ax.set_ylabel('Count', color=LABEL_COLOR)
ax.legend(title='Gender');
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()
# ============================================================
# PLOT 5: Pie Chart (Gender)
# ============================================================
print('\n--- Plot 5: Pie Chart (Gender) ---')
fig, ax = plt.subplots(figsize=(8, 8))
vc_g = df['Gender'].value_counts()
wedges, texts, autotexts = ax.pie(
        vc_g.values, labels=vc_g.index, autopct='%1.2f%%',
        colors=[C_F, C_M], startangle=90,
        textprops={'fontsize': 13}, pctdistance=0.75,
        wedgeprops={'edgecolor': 'white', 'linewidth': 2})
for t in autotexts: t.set_fontweight('bold')
ax.set_title('Gender Distribution', color=TITLE_COLOR, fontweight='bold', fontsize=16)
plt.tight_layout()
plt.show()
# ============================================================
# PLOT 6: Pie Chart (Coupon Status)
# ============================================================
print('\n--- Plot 6: Pie Chart (Coupon Status) ---')
fig, ax = plt.subplots(figsize=(8, 8))
vc_c = df['Coupon_Status'].value_counts()
wedges, texts, autotexts = ax.pie(
        vc_c.values, labels=vc_c.index, autopct='%1.2f%%',
        colors=sns.color_palette('Set2', 3), startangle=90,
        textprops={'fontsize': 13}, pctdistance=0.75,
        wedgeprops={'edgecolor': 'white', 'linewidth': 2})
for t in autotexts: t.set_fontweight('bold')
ax.set_title('Coupon Status Distribution', color=TITLE_COLOR, fontweight='bold', fontsize=16)
plt.tight_layout()
plt.show()
# ============================================================
# PLOT 7: Line Plot (Monthly Avg Spend)
# ============================================================
print('\n--- Plot 7: Line Plot (Monthly Avg Online vs Offline Spend) ---')
fig, ax = plt.subplots(figsize=(12, 6))
monthly = df.groupby('Month')[['Online_Spend', 'Offline_Spend']].mean()
ax.plot(monthly.index, monthly['Online_Spend'], 'o-', color=C_F,
        linewidth=2.5, markersize=8, label='Online Spend')
ax.plot(monthly.index, monthly['Offline_Spend'], 's--', color=C_M,
        linewidth=2.5, markersize=8, label='Offline Spend')
for i, row in monthly.iterrows():
        ax.annotate(f'{row["Online_Spend"]:.2f}', (i, row['Online_Spend']),
                    textcoords="offset points", xytext=(0, 12), fontsize=8, ha='center', color=C_F)
        ax.annotate(f'{row["Offline_Spend"]:.2f}', (i, row['Offline_Spend']),
                    textcoords="offset points", xytext=(0, -15), fontsize=8, ha='center', color=C_M)
ax.set_xlabel('Month', color=LABEL_COLOR);
ax.set_ylabel('Avg Spend ($)', color=LABEL_COLOR)
ax.set_title('Monthly Average Online vs Offline Spend', color=TITLE_COLOR, fontweight='bold')
ax.set_xticks(range(1, 13));
ax.legend();
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()
# ============================================================
# PLOT 8: Dist Plot (Online Spend)
# ============================================================
print('\n--- Plot 8: Dist Plot (Online Spend) ---')
fig, ax = plt.subplots(figsize=(10, 6))
sns.histplot(df['Online_Spend'], kde=True, color=C_M, bins=40,
             edgecolor='white', alpha=0.7, ax=ax)
mean_os = df['Online_Spend'].mean()
ax.axvline(mean_os, color=C_F, linestyle='--', linewidth=2, label=f'Mean: ${mean_os:.2f}')
ax.set_title('Distribution of Online Spend', color=TITLE_COLOR, fontweight='bold')
ax.set_xlabel('Online Spend ($)', color=LABEL_COLOR)
ax.set_ylabel('Frequency', color=LABEL_COLOR)
ax.legend();
ax.grid(alpha=0.3);
plt.tight_layout()
plt.show()
# ============================================================
# PLOT 9: Histogram + KDE (Avg Price)
# ============================================================
print('\n--- Plot 9: Histogram + KDE (Avg Price) ---')
fig, ax = plt.subplots(figsize=(10, 6))
sns.histplot(df['Avg_Price'], kde=True, bins=50, color='#2ecc71',
             edgecolor='white', alpha=0.7, stat='density', ax=ax)
m = df['Avg_Price'].mean()
ax.axvline(m, color=C_F, linestyle='--', linewidth=2, label=f'Mean: ${m:.2f}')
ax.set_title('Avg Price Distribution with KDE', color=TITLE_COLOR, fontweight='bold')
ax.set_xlabel('Avg Price ($)', color=LABEL_COLOR)
ax.set_ylabel('Density', color=LABEL_COLOR)
ax.legend();
ax.grid(alpha=0.3);
plt.tight_layout()
plt.show()
# ============================================================
# PLOT 10: KDE Plot with fill (Online Spend by Gender)
# ============================================================
print('\n--- Plot 10: KDE Plot with fill (Online Spend by Gender) ---')
fig, ax = plt.subplots(figsize=(10, 6))
sns.kdeplot(data=df, x='Online_Spend', hue='Gender', fill=True,
            alpha=0.4, palette=[C_F, C_M], linewidth=2.5, ax=ax)
ax.set_title('KDE of Online Spend by Gender', color=TITLE_COLOR, fontweight='bold')
ax.set_xlabel('Online Spend ($)', color=LABEL_COLOR)
ax.set_ylabel('Density', color=LABEL_COLOR)
ax.grid(alpha=0.3);
plt.tight_layout()
plt.show()
# ============================================================
# PLOT 11: QQ-Plot (Online Spend)
# ============================================================
print('\n--- Plot 11: QQ-Plot (Online Spend) ---')
fig, ax = plt.subplots(figsize=(8, 8))
sm.qqplot(df['Online_Spend'].dropna(), line='45', ax=ax,
          markerfacecolor=C_M, markeredgecolor='white', alpha=0.5, markersize=4)
ax.set_title('QQ-Plot of Online Spend', color=TITLE_COLOR, fontweight='bold')
ax.set_xlabel('Theoretical Quantiles', color=LABEL_COLOR)
ax.set_ylabel('Sample Quantiles', color=LABEL_COLOR)
ax.grid(alpha=0.3);
plt.tight_layout()
plt.show()
# ============================================================
# PLOT 12: Reg Plot (Online vs Offline Spend)
# ============================================================
print('\n--- Plot 12: Reg Plot (Online vs Offline Spend) ---')
fig, ax = plt.subplots(figsize=(10, 7))
samp = df.sample(3000, random_state=42)
sns.regplot(data=samp, x='Offline_Spend', y='Online_Spend',
            scatter_kws={'alpha': 0.3, 'color': C_M, 's': 15},
            line_kws={'color': C_F, 'linewidth': 2.5}, ax=ax)
r_val = df[['Online_Spend', 'Offline_Spend']].corr().iloc[0, 1]
ax.text(0.05, 0.95, f'r = {r_val:.2f}', transform=ax.transAxes,
        fontsize=14, va='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
ax.set_title('Online vs Offline Spend (Regression)', color=TITLE_COLOR, fontweight='bold')
ax.set_xlabel('Offline Spend ($)', color=LABEL_COLOR)
ax.set_ylabel('Online Spend ($)', color=LABEL_COLOR)
ax.grid(alpha=0.3);
plt.tight_layout()
plt.show()
# ============================================================
# PLOT 13: Pair Plot
# ============================================================
print('\n--- Plot 13: Pair Plot ---')
pair_cols = ['Avg_Price', 'Online_Spend', 'Offline_Spend', 'Delivery_Charges']
g = sns.pairplot(df[pair_cols + ['Gender']].sample(2000, random_state=42),
                 hue='Gender', palette=[C_F, C_M],
                 diag_kind='kde', plot_kws={'alpha': 0.4, 's': 15})
g.figure.suptitle('Pair Plot: Price, Spend & Delivery by Gender',
                  y=1.02, color=TITLE_COLOR, fontweight='bold', fontsize=16)
plt.show()
# ============================================================
# PLOT 14: Heatmap with cbar
# ============================================================
print('\n--- Plot 14: Heatmap (Pearson Correlation) ---')
fig, ax = plt.subplots(figsize=(12, 9))
corr = df[num_cols].corr().round(2)
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            mask=mask, linewidths=1, ax=ax, cbar=True,
            annot_kws={'fontsize': 10}, cbar_kws={'label': 'Pearson r'})
ax.set_title('Pearson Correlation Heatmap', color=TITLE_COLOR, fontweight='bold')
plt.xticks(rotation=45, ha='right');
plt.tight_layout()
plt.show()
# ============================================================
# PLOT 15: Multivariate Box Plot
# ============================================================
print('\n--- Plot 15: Multivariate Box Plot (Spend by Location) ---')
fig, ax = plt.subplots(figsize=(12, 6))
melt = df.melt(id_vars='Location', value_vars=['Online_Spend', 'Offline_Spend'],
               var_name='Type', value_name='Amount')
sns.boxplot(data=melt, x='Location', y='Amount', hue='Type',
            palette=[C_M, C_F], ax=ax, order=df['Location'].value_counts().index)
ax.set_title('Online vs Offline Spend by Location (Box Plot)', color=TITLE_COLOR, fontweight='bold')
ax.set_xlabel('Location', color=LABEL_COLOR)
ax.set_ylabel('Spend ($)', color=LABEL_COLOR)
ax.legend(title='Type');
ax.grid(axis='y', alpha=0.3);
plt.tight_layout()
plt.show()
# ============================================================
# PLOT 16: Multivariate Boxen Plot
# ============================================================
print('\n--- Plot 16: Boxen Plot (Online Spend by Coupon × Gender) ---')
fig, ax = plt.subplots(figsize=(10, 6))
sns.boxenplot(data=df, x='Coupon_Status', y='Online_Spend', hue='Gender',
              palette=[C_F, C_M], ax=ax)
ax.set_title('Online Spend by Coupon Status & Gender (Boxen)', color=TITLE_COLOR, fontweight='bold')
ax.set_xlabel('Coupon Status', color=LABEL_COLOR)
ax.set_ylabel('Online Spend ($)', color=LABEL_COLOR)
ax.legend(title='Gender');
ax.grid(axis='y', alpha=0.3);
plt.tight_layout()
plt.show()
# ============================================================
# PLOT 17: Area Plot (Monthly Revenue by Top 5 Categories)
# ============================================================
print('\n--- Plot 17: Area Plot (Monthly Revenue Top 5 Categories) ---')
top5 = df['Product_Category'].value_counts().head(5).index.tolist()
monthly_cat = df[df['Product_Category'].isin(top5)].groupby(
        ['Month', 'Product_Category'])['Revenue'].sum().unstack(fill_value=0)
fig, ax = plt.subplots(figsize=(12, 6))
monthly_cat.plot.area(ax=ax, alpha=0.7, colormap='Set2', linewidth=1.5)
ax.set_title('Monthly Revenue by Top 5 Categories (Area)', color=TITLE_COLOR, fontweight='bold')
ax.set_xlabel('Month', color=LABEL_COLOR)
ax.set_ylabel('Total Revenue ($)', color=LABEL_COLOR)
ax.set_xticks(range(1, 13))
ax.legend(title='Category', bbox_to_anchor=(1.05, 1), loc='upper left')
ax.grid(alpha=0.3);
plt.tight_layout()
plt.show()
# ============================================================
# PLOT 18: Violin Plot
# ============================================================
print('\n--- Plot 18: Violin Plot (Avg Price by Coupon × Gender) ---')
fig, ax = plt.subplots(figsize=(10, 6))
sns.violinplot(data=df, x='Coupon_Status', y='Avg_Price', hue='Gender',
               split=True, palette=[C_F, C_M], ax=ax, inner='quartile', cut=0)
ax.set_title('Avg Price by Coupon Status & Gender (Violin)', color=TITLE_COLOR, fontweight='bold')
ax.set_xlabel('Coupon Status', color=LABEL_COLOR)
ax.set_ylabel('Avg Price ($)', color=LABEL_COLOR)
ax.legend(title='Gender');
ax.grid(axis='y', alpha=0.3);
plt.tight_layout()
plt.show()
# ============================================================
# PLOT 19: Joint Plot (KDE + scatter)
# ============================================================
print('\n--- Plot 19: Joint Plot (Avg Price vs Delivery Charges) ---')
samp2 = df[['Avg_Price', 'Delivery_Charges']].sample(3000, random_state=42)
g = sns.jointplot(data=samp2, x='Avg_Price', y='Delivery_Charges',
                  kind='scatter', marginal_kws={'fill': True},
                  joint_kws={'alpha': 0.3, 'color': C_M, 's': 10}, height=8)
g.plot_joint(sns.kdeplot, color=C_F, levels=5, linewidths=1.5)
g.figure.suptitle('Joint Plot: Avg Price vs Delivery Charges',
                  y=1.02, color=TITLE_COLOR, fontweight='bold', fontsize=16)
plt.show()
# ============================================================
# PLOT 20: Rug Plot
# ============================================================
print('\n--- Plot 20: Rug Plot (Delivery Charges by Gender) ---')
fig, ax = plt.subplots(figsize=(10, 5))
for gender, color in zip(['F', 'M'], [C_F, C_M]):
        subset = df[df['Gender'] == gender]['Delivery_Charges'].sample(2000, random_state=42)
        sns.rugplot(data=subset, ax=ax, color=color, alpha=0.3, height=0.5, label=gender)
sns.kdeplot(data=df, x='Delivery_Charges', hue='Gender',
            palette=[C_F, C_M], ax=ax, linewidth=2, fill=False)
ax.set_title('Delivery Charges with Rug Plot', color=TITLE_COLOR, fontweight='bold')
ax.set_xlabel('Delivery Charges ($)', color=LABEL_COLOR)
ax.set_ylabel('Density', color=LABEL_COLOR)
ax.legend(title='Gender');
ax.grid(alpha=0.3);
plt.tight_layout()
plt.show()
# ============================================================
# PLOT 21: 3D Plot
# ============================================================
print('\n--- Plot 21: 3D Scatter Plot ---')
fig = plt.figure(figsize=(12, 8))
ax3 = fig.add_subplot(111, projection='3d')
s3d = df.sample(2000, random_state=42)
sc = ax3.scatter(s3d['Avg_Price'], s3d['Online_Spend'], s3d['Quantity'],
                 c=s3d['Delivery_Charges'], cmap='viridis', alpha=0.5, s=15)
ax3.set_xlabel('Avg Price ($)', labelpad=10)
ax3.set_ylabel('Online Spend ($)', labelpad=10)
ax3.set_zlabel('Quantity', labelpad=10)
ax3.set_title('3D: Price × Online Spend × Quantity (color=Delivery)',
              color=TITLE_COLOR, fontweight='bold')
fig.colorbar(sc, shrink=0.5, label='Delivery Charges ($)')
plt.tight_layout()
plt.show()
# ============================================================
# PLOT 22: Contour Plot
# ============================================================
print('\n--- Plot 22: Contour Plot (Avg Price vs Online Spend) ---')
fig, ax = plt.subplots(figsize=(10, 7))
sc2 = df[['Avg_Price', 'Online_Spend']].sample(3000, random_state=42)
sns.kdeplot(data=sc2, x='Avg_Price', y='Online_Spend',
            fill=True, cmap='Blues', levels=15, ax=ax, alpha=0.8)
sns.kdeplot(data=sc2, x='Avg_Price', y='Online_Spend',
            levels=10, ax=ax, color='navy', linewidths=1)
ax.set_title('Contour: Avg Price vs Online Spend', color=TITLE_COLOR, fontweight='bold')
ax.set_xlabel('Avg Price ($)', color=LABEL_COLOR)
ax.set_ylabel('Online Spend ($)', color=LABEL_COLOR)
ax.grid(alpha=0.3);
plt.tight_layout()
plt.show()
# ============================================================
# PLOT 23: Cluster Map
# ============================================================
print('\n--- Plot 23: Cluster Map ---')
cm = sns.clustermap(df[num_cols].corr().round(2), annot=True, fmt='.2f',
                    cmap='RdBu_r', center=0, linewidths=1,
                    figsize=(10, 10), annot_kws={'fontsize': 9})
cm.figure.suptitle('Cluster Map of Feature Correlations',
                   y=1.02, color=TITLE_COLOR, fontweight='bold', fontsize=16)
plt.show()
# ============================================================
# PLOT 24: Hexbin
# ============================================================
print('\n--- Plot 24: Hexbin (Offline vs Online Spend) ---')
fig, ax = plt.subplots(figsize=(10, 7))
hb = ax.hexbin(df['Offline_Spend'], df['Online_Spend'], gridsize=30,
               cmap='YlOrRd', mincnt=1)
ax.set_title('Hexbin: Offline vs Online Spend', color=TITLE_COLOR, fontweight='bold')
ax.set_xlabel('Offline Spend ($)', color=LABEL_COLOR)
ax.set_ylabel('Online Spend ($)', color=LABEL_COLOR)
fig.colorbar(hb, ax=ax, label='Count')
ax.grid(alpha=0.3);
plt.tight_layout()
plt.show()
# ============================================================
# PLOT 25: Strip Plot
# ============================================================
print('\n--- Plot 25: Strip Plot (Avg Price by Top 6 Categories) ---')
fig, ax = plt.subplots(figsize=(12, 7))
top6 = df['Product_Category'].value_counts().head(6).index.tolist()
sns.stripplot(data=df[df['Product_Category'].isin(top6)],
              x='Product_Category', y='Avg_Price', hue='Gender',
              palette=[C_F, C_M], alpha=0.3, size=3, dodge=True, ax=ax, order=top6)
ax.set_title('Strip: Avg Price by Top 6 Categories & Gender', color=TITLE_COLOR, fontweight='bold')
ax.set_xlabel('Category', color=LABEL_COLOR)
ax.set_ylabel('Avg Price ($)', color=LABEL_COLOR)
ax.legend(title='Gender');
ax.grid(axis='y', alpha=0.3)
plt.xticks(rotation=20);
plt.tight_layout()
plt.show()
# ============================================================
# PLOT 26: Swarm Plot
# ============================================================
print('\n--- Plot 26: Swarm Plot (Delivery Charges by Location) ---')
fig, ax = plt.subplots(figsize=(10, 6))
sw = df.sample(800, random_state=42)  # small sample — swarm is slow
sns.swarmplot(data=sw, x='Location', y='Delivery_Charges', hue='Gender',
              palette=[C_F, C_M], size=3, ax=ax,
              order=df['Location'].value_counts().index)
ax.set_title('Swarm: Delivery Charges by Location & Gender', color=TITLE_COLOR, fontweight='bold')
ax.set_xlabel('Location', color=LABEL_COLOR)
ax.set_ylabel('Delivery Charges ($)', color=LABEL_COLOR)
ax.legend(title='Gender');
ax.grid(axis='y', alpha=0.3);
plt.tight_layout()
plt.show()
# ============================================================
# SUBPLOTS (4+) — STORYTELLING
# ============================================================
print('\n' + '=' * 60)
print('SUBPLOTS — STORYTELLING')
print('=' * 60)

# --- Subplot A: 4 Pie Charts — Category breakdown by Location ---
print('\n--- Subplot A: 4 Pie Charts (Top Categories by Location) ---')
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
fig.suptitle('Top 5 Product Categories by Location', color=TITLE_COLOR,
             fontweight='bold', fontsize=18, y=1.01)
top5cat = df['Product_Category'].value_counts().head(5).index
locs = ['Chicago', 'California', 'New York', 'New Jersey']
for ax, loc in zip(axes.flatten(), locs):
        sub = df[df['Location'] == loc]['Product_Category'].value_counts()
        sub_top = sub[sub.index.isin(top5cat)]
        sub_top['Other'] = sub[~sub.index.isin(top5cat)].sum()
        ax.pie(sub_top.values, labels=sub_top.index, autopct='%1.1f%%',
               colors=sns.color_palette('Set2', len(sub_top)),
               textprops={'fontsize': 9}, wedgeprops={'edgecolor': 'white'})
        ax.set_title(loc, fontsize=14, fontweight='bold', color=LABEL_COLOR)
plt.tight_layout()
plt.show()

# --- Subplot B: 4 Bar Charts — Spend patterns ---
print('\n--- Subplot B: 4 Bar Charts (Spend Patterns) ---')
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Spending Patterns Analysis', color=TITLE_COLOR,
             fontweight='bold', fontsize=18, y=1.01)
plt.show()
# B1: Avg Online Spend by Gender
ax = axes[0, 0]
g_spend = df.groupby('Gender')['Online_Spend'].mean().sort_values(ascending=False)
bars = ax.bar(g_spend.index, g_spend.values, color=[C_F, C_M], edgecolor='white')
for b in bars: ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 20,
                       f'${b.get_height():.2f}', ha='center', fontsize=10)
ax.set_title('Avg Online Spend by Gender', color=LABEL_COLOR, fontweight='bold')
ax.set_ylabel('Avg Spend ($)');
ax.grid(axis='y', alpha=0.3)
plt.show()
# B2: Avg Online Spend by Location
ax = axes[0, 1]
l_spend = df.groupby('Location')['Online_Spend'].mean().sort_values(ascending=False)
bars = ax.bar(l_spend.index, l_spend.values, color=sns.color_palette('Set2', 5), edgecolor='white')
for b in bars: ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 20,
                       f'${b.get_height():.2f}', ha='center', fontsize=9)
ax.set_title('Avg Online Spend by Location', color=LABEL_COLOR, fontweight='bold')
ax.set_ylabel('Avg Spend ($)');
ax.grid(axis='y', alpha=0.3)
plt.setp(ax.get_xticklabels(), rotation=20)
plt.show()
# B3: Avg Online Spend by Coupon Status
ax = axes[1, 0]
c_spend = df.groupby('Coupon_Status')['Online_Spend'].mean().sort_values(ascending=False)
bars = ax.bar(c_spend.index, c_spend.values, color=sns.color_palette('Set2', 3), edgecolor='white')
for b in bars: ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 20,
                       f'${b.get_height():.2f}', ha='center', fontsize=10)
ax.set_title('Avg Online Spend by Coupon Status', color=LABEL_COLOR, fontweight='bold')
ax.set_ylabel('Avg Spend ($)');
ax.grid(axis='y', alpha=0.3)
plt.show()
# B4: Avg Online Spend by Top 5 Category
ax = axes[1, 1]
cat_spend = df.groupby('Product_Category')['Online_Spend'].mean().sort_values(ascending=False).head(5)
bars = ax.barh(cat_spend.index, cat_spend.values, color=sns.color_palette('Set2', 5), edgecolor='white')
for b, v in zip(bars, cat_spend.values):
        ax.text(v + 10, b.get_y() + b.get_height() / 2, f'${v:.2f}', va='center', fontsize=10)
ax.set_title('Avg Online Spend by Top 5 Category', color=LABEL_COLOR, fontweight='bold')
ax.set_xlabel('Avg Spend ($)');
ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.show()
# --- Subplot C: Histograms for multiple numerical features ---
print('\n--- Subplot C: 4 Histograms (Numerical Feature Distributions) ---')
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Distributions of Key Numerical Features', color=TITLE_COLOR,
             fontweight='bold', fontsize=18, y=1.01)
feats = ['Avg_Price', 'Delivery_Charges', 'Online_Spend', 'Offline_Spend']
colors = ['#2ecc71', '#9b59b6', '#3498db', '#e74c3c']
for ax, feat, col in zip(axes.flatten(), feats, colors):
        sns.histplot(df[feat], kde=True, bins=40, color=col, edgecolor='white', alpha=0.7, ax=ax)
        mu = df[feat].mean()
        ax.axvline(mu, color='black', linestyle='--', linewidth=1.5, label=f'Mean: {mu:.2f}')
        ax.set_title(feat, color=LABEL_COLOR, fontweight='bold')
        ax.legend(fontsize=9);
        ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()
# --- Subplot D: KDE comparison across locations ---
print('\n--- Subplot D: KDE of Online Spend by Location (4 panels) ---')
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Online Spend Distribution by Location', color=TITLE_COLOR,
             fontweight='bold', fontsize=18, y=1.01)
for ax, loc in zip(axes.flatten(), ['Chicago', 'California', 'New York', 'New Jersey']):
        sub = df[df['Location'] == loc]
        sns.kdeplot(data=sub, x='Online_Spend', hue='Gender', fill=True,
                    palette=[C_F, C_M], alpha=0.4, linewidth=2, ax=ax)
        ax.set_title(loc, color=LABEL_COLOR, fontweight='bold')
        ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# ============================================================
# ============================================================
# SECTION A: STATISTICS — Descriptive & Inferential
# ============================================================
# ============================================================
print('\n' + '=' * 60)
print('SECTION A: STATISTICS')
print('=' * 60)

from scipy.stats import skew, kurtosis, ttest_ind, f_oneway
from scipy.stats import gaussian_kde

# --- A1: Descriptive Statistics Table (PrettyTable) ---
print('\n--- A1: Descriptive Statistics Table ---')
t_stat = PrettyTable()
t_stat.title = 'Descriptive Statistics of Numerical Features'
t_stat.field_names = ['Feature', 'Mean', 'Std', 'Min', 'Max', 'Median', 'Skewness', 'Kurtosis']
for col in num_cols:
    t_stat.add_row([
        col,
        f'{df[col].mean():.2f}',
        f'{df[col].std():.2f}',
        f'{df[col].min():.2f}',
        f'{df[col].max():.2f}',
        f'{df[col].median():.2f}',
        f'{skew(df[col].dropna()):.2f}',
        f'{kurtosis(df[col].dropna()):.2f}'
    ])
print(t_stat)

# --- A2: T-Test — Online Spend by Gender ---
print('\n--- A2: T-Test (Online Spend: Female vs Male) ---')
grp_f = df[df['Gender'] == 'F']['Online_Spend']
grp_m = df[df['Gender'] == 'M']['Online_Spend']
t_stat_val, p_val = ttest_ind(grp_f, grp_m, equal_var=False)
t_ttest = PrettyTable()
t_ttest.title = 'Welch T-Test: Online Spend (F vs M)'
t_ttest.field_names = ['Group', 'Mean', 'Std', 'N']
t_ttest.add_row(['Female', f'{grp_f.mean():.2f}', f'{grp_f.std():.2f}', f'{len(grp_f):,}'])
t_ttest.add_row(['Male',   f'{grp_m.mean():.2f}', f'{grp_m.std():.2f}', f'{len(grp_m):,}'])
print(t_ttest)
print(f'  t-statistic: {t_stat_val:.4f}  |  p-value: {p_val:.4f}')
print('  → Conclusion:', 'Significant difference (p<0.05)' if p_val < 0.05 else 'No significant difference')

# --- A3: One-Way ANOVA — Online Spend by Location ---
print('\n--- A3: One-Way ANOVA (Online Spend by Location) ---')
groups = [grp['Online_Spend'].values for _, grp in df.groupby('Location')]
f_stat_val, p_anova = f_oneway(*groups)
t_anova = PrettyTable()
t_anova.title = 'One-Way ANOVA: Online Spend by Location'
t_anova.field_names = ['Location', 'Mean', 'Std', 'N']
for loc, grp in df.groupby('Location'):
    t_anova.add_row([loc, f'{grp["Online_Spend"].mean():.2f}',
                     f'{grp["Online_Spend"].std():.2f}', f'{len(grp):,}'])
print(t_anova)
print(f'  F-statistic: {f_stat_val:.4f}  |  p-value: {p_anova:.4f}')
print('  → Conclusion:', 'Significant difference across locations (p<0.05)' if p_anova < 0.05 else 'No significant difference')

# --- A4: Multivariate KDE Plot ---
print('\n--- A4: Multivariate KDE Estimate (Online vs Offline Spend) ---')
fig, ax = plt.subplots(figsize=(10, 7))
samp_kde = df[['Online_Spend', 'Offline_Spend']].sample(3000, random_state=42)
sns.kdeplot(data=samp_kde, x='Offline_Spend', y='Online_Spend',
            fill=True, cmap='mako', levels=15, alpha=0.8, ax=ax)
sns.kdeplot(data=samp_kde, x='Offline_Spend', y='Online_Spend',
            levels=10, ax=ax, color='white', linewidths=0.8, alpha=0.6)
ax.set_title('Multivariate KDE: Online vs Offline Spend', color=TITLE_COLOR, fontweight='bold')
ax.set_xlabel('Offline Spend ($)', color=LABEL_COLOR)
ax.set_ylabel('Online Spend ($)', color=LABEL_COLOR)
ax.grid(alpha=0.2)
plt.tight_layout()
plt.show()
print('Observation: The multivariate KDE reveals the joint density of Online and Offline Spend.')
print('The highest density region indicates where most customers spend similar amounts in both channels.')

# ============================================================
# ============================================================
# SECTION B: OUTLIER DETECTION & REMOVAL
# ============================================================
# ============================================================
print('\n' + '=' * 60)
print('SECTION B: OUTLIER DETECTION & REMOVAL')
print('=' * 60)

from scipy.stats import zscore

df_clean = df.copy()
n_before = len(df_clean)

# --- B1: IQR Method ---
print('\n--- B1: IQR-Based Outlier Detection ---')
t_iqr = PrettyTable()
t_iqr.title = 'Outlier Detection via IQR Method'
t_iqr.field_names = ['Feature', 'Q1', 'Q3', 'IQR', 'Lower Fence', 'Upper Fence', 'Outliers']
outlier_mask = pd.Series([False] * len(df_clean), index=df_clean.index)
for col in ['Online_Spend', 'Offline_Spend', 'Avg_Price', 'Delivery_Charges', 'Revenue']:
    Q1 = df_clean[col].quantile(0.25)
    Q3 = df_clean[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    mask = (df_clean[col] < lower) | (df_clean[col] > upper)
    outlier_mask = outlier_mask | mask
    t_iqr.add_row([col, f'{Q1:.2f}', f'{Q3:.2f}', f'{IQR:.2f}',
                   f'{lower:.2f}', f'{upper:.2f}', f'{mask.sum():,}'])
print(t_iqr)

# --- B2: Z-Score Method (for comparison) ---
print('\n--- B2: Z-Score Outlier Detection (|z| > 3) ---')
t_zscore = PrettyTable()
t_zscore.title = 'Outlier Detection via Z-Score (threshold=3)'
t_zscore.field_names = ['Feature', 'Mean', 'Std', 'Z>3 Outliers', '% of Data']
for col in ['Online_Spend', 'Offline_Spend', 'Avg_Price', 'Delivery_Charges', 'Revenue']:
    z = np.abs(zscore(df_clean[col].dropna()))
    n_out = (z > 3).sum()
    t_zscore.add_row([col, f'{df_clean[col].mean():.2f}', f'{df_clean[col].std():.2f}',
                      f'{n_out:,}', f'{100*n_out/len(df_clean):.2f}%'])
print(t_zscore)

# Apply IQR removal
df_no_outlier = df_clean[~outlier_mask].copy()
n_after = len(df_no_outlier)
print(f'\n  Rows before IQR removal : {n_before:,}')
print(f'  Rows after  IQR removal : {n_after:,}')
print(f'  Removed                 : {n_before - n_after:,} ({100*(n_before-n_after)/n_before:.2f}%)')

# --- B3: Box Plot Before vs After ---
print('\n--- B3: Box Plot — Before vs After Outlier Removal ---')
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Outlier Removal Comparison (IQR Method)', color=TITLE_COLOR,
             fontweight='bold', fontsize=16)
for ax, data, label in zip(axes, [df, df_no_outlier], ['Before', 'After']):
    melt_b = data.melt(value_vars=['Online_Spend', 'Offline_Spend', 'Avg_Price', 'Revenue'],
                       var_name='Feature', value_name='Value')
    sns.boxplot(data=melt_b, x='Feature', y='Value', palette='Set2', ax=ax)
    ax.set_title(f'{label} Outlier Removal', color=LABEL_COLOR, fontweight='bold')
    ax.set_xlabel('Feature', color=LABEL_COLOR)
    ax.set_ylabel('Value ($)', color=LABEL_COLOR)
    ax.grid(axis='y', alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=15)
plt.tight_layout()
plt.show()
print('Observation: IQR method effectively removes extreme values.')
print('Revenue and Online_Spend show the most outliers due to high-value transactions.')

# ============================================================
# ============================================================
# SECTION C: NORMALITY TEST
# ============================================================
# ============================================================
print('\n' + '=' * 60)
print('SECTION C: NORMALITY TESTS')
print('=' * 60)

from scipy.stats import shapiro, normaltest, anderson

# --- C1: Shapiro-Wilk, D'Agostino Tests ---
print('\n--- C1: Normality Test Results Table ---')
t_norm = PrettyTable()
t_norm.title = 'Normality Tests (Shapiro-Wilk & D\'Agostino K²)'
t_norm.field_names = ['Feature', 'Shapiro W', 'Shapiro p', 'SW Normal?',
                       'D\'Agostino p', 'DA Normal?']
test_cols = ['Online_Spend', 'Offline_Spend', 'Avg_Price', 'Delivery_Charges', 'GST']
for col in test_cols:
    samp_n = df[col].dropna().sample(min(5000, len(df)), random_state=42)
    sw_stat, sw_p = shapiro(samp_n.sample(min(5000, len(samp_n)), random_state=42))
    da_stat, da_p = normaltest(samp_n)
    t_norm.add_row([col,
                    f'{sw_stat:.4f}', f'{sw_p:.4f}', 'Yes' if sw_p > 0.05 else 'No',
                    f'{da_p:.4f}',    'Yes' if da_p > 0.05 else 'No'])
print(t_norm)
print('\nObservation: Most financial variables (Online_Spend, Offline_Spend) are NOT normally distributed')
print('(p < 0.05), indicating right-skewed distributions typical of transaction data.')

# --- C2: QQ plots for 4 features ---
print('\n--- C2: QQ Plots for Multiple Features ---')
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('QQ-Plots: Normality Assessment', color=TITLE_COLOR,
             fontweight='bold', fontsize=18, y=1.01)
qq_feats = ['Online_Spend', 'Offline_Spend', 'Avg_Price', 'Delivery_Charges']
colors_qq = [C_M, C_F, '#2ecc71', '#9b59b6']
for ax, feat, col in zip(axes.flatten(), qq_feats, colors_qq):
    sm.qqplot(df[feat].dropna(), line='45', ax=ax,
              markerfacecolor=col, markeredgecolor='white', alpha=0.4, markersize=3)
    ax.set_title(feat, color=LABEL_COLOR, fontweight='bold')
    ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# ============================================================
# ============================================================
# SECTION D: DATA TRANSFORMATION
# ============================================================
# ============================================================
print('\n' + '=' * 60)
print('SECTION D: DATA TRANSFORMATION')
print('=' * 60)

from sklearn.preprocessing import StandardScaler, MinMaxScaler

# --- D1: Log Transformation ---
print('\n--- D1: Log Transformation (Online_Spend) ---')
df['Online_Spend_log'] = np.log1p(df['Online_Spend'])
df['Offline_Spend_log'] = np.log1p(df['Offline_Spend'])
df['Revenue_log'] = np.log1p(df['Revenue'])

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.suptitle('Log Transformation: Before vs After', color=TITLE_COLOR,
             fontweight='bold', fontsize=16)
pairs = [('Online_Spend', 'Online_Spend_log'),
         ('Offline_Spend', 'Offline_Spend_log'),
         ('Revenue', 'Revenue_log')]
for i, (orig, log_c) in enumerate(pairs):
    for j, (col, label) in enumerate([(orig, 'Original'), (log_c, 'Log Transformed')]):
        ax = axes[j][i]
        sk = skew(df[col].dropna())
        sns.histplot(df[col], kde=True, bins=40, ax=ax,
                     color=['#3498db', '#e74c3c'][j], alpha=0.7, edgecolor='white')
        ax.set_title(f'{label}: {orig}\n(Skew={sk:.2f})', color=LABEL_COLOR, fontweight='bold')
        ax.set_xlabel(col, color=LABEL_COLOR)
        ax.set_ylabel('Frequency', color=LABEL_COLOR)
        ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# --- D2: Normalization Table ---
print('\n--- D2: Normalization Comparison Table ---')
t_trans = PrettyTable()
t_trans.title = 'Before vs After Transformation (Online_Spend)'
t_trans.field_names = ['Metric', 'Original', 'Log1p', 'Z-Score', 'MinMax']
sc = StandardScaler().fit_transform(df[['Online_Spend']])
mm = MinMaxScaler().fit_transform(df[['Online_Spend']])
data_z = pd.Series(sc.flatten())
data_mm = pd.Series(mm.flatten())
data_log = df['Online_Spend_log']
orig = df['Online_Spend']
for metric, fn in [('Mean', lambda x: x.mean()), ('Std', lambda x: x.std()),
                   ('Skewness', lambda x: skew(x.dropna())), ('Min', lambda x: x.min()),
                   ('Max', lambda x: x.max())]:
    t_trans.add_row([metric, f'{fn(orig):.2f}', f'{fn(data_log):.2f}',
                     f'{fn(data_z):.2f}', f'{fn(data_mm):.2f}'])
print(t_trans)
print('\nObservation: Log transformation reduces skewness from highly right-skewed to near-normal,')
print('making the data more suitable for statistical modeling and linear methods.')

# ============================================================
# ============================================================
# SECTION E: PRINCIPAL COMPONENT ANALYSIS (PCA)
# ============================================================
# ============================================================
print('\n' + '=' * 60)
print('SECTION E: PRINCIPAL COMPONENT ANALYSIS (PCA)')
print('=' * 60)

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Standardize
pca_cols = ['Tenure_Months', 'Quantity', 'Avg_Price', 'Delivery_Charges',
            'GST', 'Offline_Spend', 'Online_Spend', 'Discount_pct', 'Total_Spend', 'Revenue']
df_pca_data = df[pca_cols].dropna()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_pca_data)

# Full PCA
pca_full = PCA()
pca_full.fit(X_scaled)

# --- E1: Explained Variance Table ---
print('\n--- E1: PCA Explained Variance Table ---')
t_pca = PrettyTable()
t_pca.title = 'PCA Explained Variance'
t_pca.field_names = ['PC', 'Eigenvalue', 'Explained Var %', 'Cumulative %']
cum = 0
for i, (ev, evr) in enumerate(zip(pca_full.explained_variance_,
                                   pca_full.explained_variance_ratio_)):
    cum += evr * 100
    t_pca.add_row([f'PC{i+1}', f'{ev:.4f}', f'{evr*100:.2f}%', f'{cum:.2f}%'])
print(t_pca)

# --- E2: Condition Number & Singular Values ---
print('\n--- E2: Condition Number & Singular Values ---')
cond_num = np.linalg.cond(X_scaled)
svd_vals = pca_full.singular_values_
t_svd = PrettyTable()
t_svd.title = 'Singular Values of Standardized Feature Matrix'
t_svd.field_names = ['PC'] + [f'PC{i+1}' for i in range(len(svd_vals))]
t_svd.add_row(['Singular Value'] + [f'{v:.2f}' for v in svd_vals])
print(t_svd)
print(f'\n  Condition Number: {cond_num:.2f}')
print('  → A large condition number suggests multicollinearity among features.')

# --- E3: Scree Plot ---
print('\n--- E3: Scree Plot + Cumulative Variance ---')
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('PCA Analysis: Online Shopping Dataset', color=TITLE_COLOR,
             fontweight='bold', fontsize=16)
ax1 = axes[0]
evr = pca_full.explained_variance_ratio_ * 100
bars = ax1.bar(range(1, len(evr)+1), evr, color=sns.color_palette('Set2', len(evr)), edgecolor='white')
for b, v in zip(bars, evr):
    ax1.text(b.get_x()+b.get_width()/2, b.get_height()+0.3, f'{v:.1f}%',
             ha='center', fontsize=9)
ax1.set_title('Scree Plot (Explained Variance per PC)', color=LABEL_COLOR, fontweight='bold')
ax1.set_xlabel('Principal Component', color=LABEL_COLOR)
ax1.set_ylabel('Explained Variance (%)', color=LABEL_COLOR)
ax1.grid(axis='y', alpha=0.3)
ax2 = axes[1]
cum_var = np.cumsum(evr)
ax2.plot(range(1, len(cum_var)+1), cum_var, 'o-', color=C_F, linewidth=2.5, markersize=8)
ax2.axhline(95, color=C_M, linestyle='--', linewidth=1.5, label='95% threshold')
ax2.axhline(80, color='green', linestyle='--', linewidth=1.5, label='80% threshold')
for i, v in enumerate(cum_var):
    ax2.annotate(f'{v:.1f}%', (i+1, v), textcoords='offset points',
                 xytext=(0, 8), fontsize=8, ha='center', color=TITLE_COLOR)
ax2.set_title('Cumulative Explained Variance', color=LABEL_COLOR, fontweight='bold')
ax2.set_xlabel('Number of Components', color=LABEL_COLOR)
ax2.set_ylabel('Cumulative Variance (%)', color=LABEL_COLOR)
ax2.legend(); ax2.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# --- E4: PCA Biplot (PC1 vs PC2 with loadings) ---
print('\n--- E4: PCA Biplot (PC1 vs PC2) ---')
pca2 = PCA(n_components=2)
X_pca2 = pca2.fit_transform(X_scaled)
samp_idx = np.random.choice(len(X_pca2), 2000, replace=False)
fig, ax = plt.subplots(figsize=(12, 8))
scatter = ax.scatter(X_pca2[samp_idx, 0], X_pca2[samp_idx, 1],
                     c=df_pca_data.iloc[samp_idx]['Online_Spend'],
                     cmap='viridis', alpha=0.4, s=10)
plt.colorbar(scatter, ax=ax, label='Online Spend ($)')
scale = 3
for i, feat in enumerate(pca_cols):
    ax.arrow(0, 0,
             pca2.components_[0, i]*scale,
             pca2.components_[1, i]*scale,
             head_width=0.08, head_length=0.05, fc=C_F, ec=C_F)
    ax.text(pca2.components_[0, i]*scale*1.15,
            pca2.components_[1, i]*scale*1.15,
            feat, fontsize=9, color=TITLE_COLOR, fontweight='bold')
ax.set_title(f'PCA Biplot (PC1={pca2.explained_variance_ratio_[0]*100:.1f}% | '
             f'PC2={pca2.explained_variance_ratio_[1]*100:.1f}%)',
             color=TITLE_COLOR, fontweight='bold')
ax.set_xlabel('PC1', color=LABEL_COLOR); ax.set_ylabel('PC2', color=LABEL_COLOR)
ax.axhline(0, color='gray', linewidth=0.5); ax.axvline(0, color='gray', linewidth=0.5)
ax.grid(alpha=0.2)
plt.tight_layout()
plt.show()
print('Observation: PC1 is dominated by spend-related features (Total_Spend, Online/Offline_Spend, Revenue),')
print('while PC2 separates Delivery_Charges and Discount_pct. ~2-3 PCs explain most variance.')

# ============================================================
# ============================================================
# SECTION F: ADDITIONAL STATISTICS & PRETTY TABLES
# ============================================================
# ============================================================
print('\n' + '=' * 60)
print('SECTION F: ADDITIONAL TABLES')
print('=' * 60)

# --- F1: Dataset Summary Table ---
print('\n--- F1: Dataset Summary ---')
t_summary = PrettyTable()
t_summary.title = 'Dataset Summary'
t_summary.field_names = ['Attribute', 'Value']
t_summary.add_row(['Total Rows',         f'{len(df):,}'])
t_summary.add_row(['Total Columns',       f'{df.shape[1]}'])
t_summary.add_row(['Numerical Features',  f'{len(num_cols)}'])
t_summary.add_row(['Categorical Features',f'{len(cat_cols)}'])
t_summary.add_row(['Missing Values',      f'{df.isnull().sum().sum()}'])
t_summary.add_row(['Duplicate Rows',      f'{df.duplicated().sum():,}'])
t_summary.add_row(['Date Range',          f'{df["Transaction_Date"].min().date()} → {df["Transaction_Date"].max().date()}'])
t_summary.add_row(['Unique Customers',    f'{df["CustomerID"].nunique():,}'])
t_summary.add_row(['Unique Products',     f'{df["Product_Category"].nunique()}'])
print(t_summary)

# --- F2: Category Distribution Table ---
print('\n--- F2: Categorical Feature Distributions ---')
for col in cat_cols:
    t_cat = PrettyTable()
    t_cat.title = f'Distribution: {col}'
    t_cat.field_names = [col, 'Count', 'Percentage']
    vc = df[col].value_counts()
    for val, cnt in vc.items():
        t_cat.add_row([val, f'{cnt:,}', f'{100*cnt/len(df):.2f}%'])
    print(t_cat)

# --- F3: Pearson Correlation Table (expanded) ---
print('\n--- F3: Full Pearson Correlation Table ---')
corr_full = df[num_cols].corr().round(2)
t_corr = PrettyTable()
t_corr.title = 'Pearson Correlation Matrix'
t_corr.field_names = [''] + num_cols
for idx in corr_full.index:
    t_corr.add_row([idx] + [f'{v:.2f}' for v in corr_full.loc[idx]])
print(t_corr)

print('\n' + '=' * 60)
print('ALL PHASE I ANALYSES COMPLETE')
print('=' * 60)

# ============================================================
# ============================================================
# PROPOSAL INSIGHT PLOTS (추가 4개)
# ============================================================
# ============================================================
print('\n' + '=' * 60)
print('PROPOSAL INSIGHT PLOTS')
print('=' * 60)

# ============================================================
# PROPOSAL PLOT 1: GST vs Avg_Price — Scatter + Regression
# ============================================================
print('\n--- Proposal Plot 1: GST vs Avg_Price (Negative Correlation r=-0.58) ---')
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle('GST vs Avg Price: Negative Correlation Analysis',
             color=TITLE_COLOR, fontweight='bold', fontsize=16)

# Left: Scatter + Regression by Gender
ax = axes[0]
samp_gst = df.sample(3000, random_state=42)
sns.regplot(data=samp_gst, x='Avg_Price', y='GST',
            scatter_kws={'alpha': 0.25, 'color': C_M, 's': 12},
            line_kws={'color': C_F, 'linewidth': 2.5}, ax=ax)
r_gst = df[['Avg_Price', 'GST']].corr().iloc[0, 1]
ax.text(0.05, 0.95, f'r = {r_gst:.2f}', transform=ax.transAxes,
        fontsize=14, va='top', fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='#fadbd8', alpha=0.8))
ax.set_title('GST vs Avg Price (Regression)', color=LABEL_COLOR, fontweight='bold')
ax.set_xlabel('Avg Price ($)', color=LABEL_COLOR)
ax.set_ylabel('GST (%)', color=LABEL_COLOR)
ax.grid(alpha=0.3)

# Right: GST by Price Bin (bar)
ax = axes[1]
df['Price_Bin'] = pd.cut(df['Avg_Price'],
                          bins=[0, 25, 50, 100, 200, 400],
                          labels=['$0-25', '$25-50', '$50-100', '$100-200', '$200+'])
gst_by_price = df.groupby('Price_Bin', observed=True)['GST'].mean()
bars = ax.bar(gst_by_price.index, gst_by_price.values,
              color=sns.color_palette('RdYlGn_r', len(gst_by_price)), edgecolor='white')
for b, v in zip(bars, gst_by_price.values):
    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.01,
            f'{v:.2f}%', ha='center', fontsize=11, fontweight='bold', color=TITLE_COLOR)
ax.set_title('Avg GST Rate by Price Bracket', color=LABEL_COLOR, fontweight='bold')
ax.set_xlabel('Price Range', color=LABEL_COLOR)
ax.set_ylabel('Mean GST (%)', color=LABEL_COLOR)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()
print('Observation: GST shows a strong negative correlation with Avg_Price (r=-0.58).')
print('Lower-priced items carry higher GST rates, while premium products ($200+)')
print('have significantly lower effective GST — suggesting tiered tax structures.')

# ============================================================
# PROPOSAL PLOT 2: Online vs Offline Spend — Distribution Comparison
# ============================================================
print('\n--- Proposal Plot 2: Online vs Offline Spend Distribution Comparison ---')
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Online vs Offline Spend: Distribution Deep Dive',
             color=TITLE_COLOR, fontweight='bold', fontsize=16)

# Left: KDE Overlay
ax = axes[0]
sns.kdeplot(df['Online_Spend'], ax=ax, fill=True, alpha=0.45,
            color=C_M, linewidth=2.5, label='Online Spend')
sns.kdeplot(df['Offline_Spend'], ax=ax, fill=True, alpha=0.45,
            color=C_F, linewidth=2.5, label='Offline Spend')
ax.axvline(df['Online_Spend'].mean(), color=C_M, linestyle='--',
           linewidth=1.8, label=f'Online Mean: ${df["Online_Spend"].mean():.2f}')
ax.axvline(df['Offline_Spend'].mean(), color=C_F, linestyle='--',
           linewidth=1.8, label=f'Offline Mean: ${df["Offline_Spend"].mean():.2f}')
ax.set_title('KDE Overlay: Online vs Offline Spend', color=LABEL_COLOR, fontweight='bold')
ax.set_xlabel('Spend ($)', color=LABEL_COLOR)
ax.set_ylabel('Density', color=LABEL_COLOR)
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

# Middle: Histogram comparison (side by side bins)
ax = axes[1]
ax.hist(df['Online_Spend'], bins=40, alpha=0.6, color=C_M,
        edgecolor='white', label='Online', density=True)
ax.hist(df['Offline_Spend'], bins=40, alpha=0.6, color=C_F,
        edgecolor='white', label='Offline', density=True)
ax.set_title('Histogram: Spend Distribution Shape', color=LABEL_COLOR, fontweight='bold')
ax.set_xlabel('Spend ($)', color=LABEL_COLOR)
ax.set_ylabel('Density', color=LABEL_COLOR)
ax.legend()
ax.grid(alpha=0.3)

# Right: Stats comparison table as text
ax = axes[2]
ax.axis('off')
stats_data = {
    'Metric': ['Mean', 'Median', 'Std Dev', 'Min', 'Max', 'Skewness'],
    'Online ($)': [
        f'{df["Online_Spend"].mean():.2f}',
        f'{df["Online_Spend"].median():.2f}',
        f'{df["Online_Spend"].std():.2f}',
        f'{df["Online_Spend"].min():.2f}',
        f'{df["Online_Spend"].max():.2f}',
        f'{stats.skew(df["Online_Spend"]):.2f}'
    ],
    'Offline ($)': [
        f'{df["Offline_Spend"].mean():.2f}',
        f'{df["Offline_Spend"].median():.2f}',
        f'{df["Offline_Spend"].std():.2f}',
        f'{df["Offline_Spend"].min():.2f}',
        f'{df["Offline_Spend"].max():.2f}',
        f'{stats.skew(df["Offline_Spend"]):.2f}'
    ]
}
tbl = ax.table(
    cellText=list(zip(stats_data['Metric'],
                      stats_data['Online ($)'],
                      stats_data['Offline ($)'])),
    colLabels=['Metric', 'Online ($)', 'Offline ($)'],
    cellLoc='center', loc='center',
    bbox=[0, 0.1, 1, 0.85]
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(11)
for (r, c), cell in tbl.get_celld().items():
    if r == 0:
        cell.set_facecolor('#2c3e50')
        cell.set_text_props(color='white', fontweight='bold')
    elif c == 1:
        cell.set_facecolor('#d6eaf8')
    elif c == 2:
        cell.set_facecolor('#fadbd8')
    cell.set_edgecolor('white')
ax.set_title('Summary Statistics Comparison', color=LABEL_COLOR, fontweight='bold')

plt.tight_layout()
plt.show()
print('Observation: Online Spend shows a near-uniform distribution ($320-$4,557)')
print('while Offline Spend has a higher median and wider IQR, confirming customers')
print('spend more in physical stores but online spend is more evenly distributed.')

# ============================================================
# PROPOSAL PLOT 3: Category Price Range Box Plot
# (프로포잘: Nest-USA 가격 범위 $0-$350으로 가장 넓음)
# ============================================================
print('\n--- Proposal Plot 3: Category Price Range Box Plot (Nest-USA Highlight) ---')
fig, axes = plt.subplots(1, 2, figsize=(18, 7))
fig.suptitle('Product Category Price Range Analysis',
             color=TITLE_COLOR, fontweight='bold', fontsize=16)

# Left: Box plot all categories sorted by median price
ax = axes[0]
cat_order = (df.groupby('Product_Category')['Avg_Price']
               .median()
               .sort_values(ascending=False)
               .index.tolist())
palette_cat = ['#e74c3c' if c == 'Nest-USA' else '#3498db' for c in cat_order]
sns.boxplot(data=df, y='Product_Category', x='Avg_Price',
            order=cat_order, palette=palette_cat, ax=ax,
            flierprops=dict(marker='o', markerfacecolor='gray',
                            markersize=3, alpha=0.3))
ax.set_title('Price Range by Category (sorted by median)\n🔴 = Nest-USA',
             color=LABEL_COLOR, fontweight='bold')
ax.set_xlabel('Avg Price ($)', color=LABEL_COLOR)
ax.set_ylabel('Product Category', color=LABEL_COLOR)
ax.grid(axis='x', alpha=0.3)
ax.axvline(df['Avg_Price'].mean(), color='green', linestyle='--',
           linewidth=1.5, label=f'Overall Mean: ${df["Avg_Price"].mean():.2f}')
ax.legend(fontsize=9)

# Right: IQR + Range bar chart per category
ax = axes[1]
cat_stats = df.groupby('Product_Category')['Avg_Price'].agg(
    Q1=lambda x: x.quantile(0.25),
    Q3=lambda x: x.quantile(0.75),
    median='median',
    price_max='max'
).reset_index()
cat_stats['IQR'] = cat_stats['Q3'] - cat_stats['Q1']
cat_stats = cat_stats.sort_values('IQR', ascending=True)
colors_iqr = ['#e74c3c' if c == 'Nest-USA' else '#85B7EB'
              for c in cat_stats['Product_Category']]
bars = ax.barh(cat_stats['Product_Category'], cat_stats['IQR'],
               color=colors_iqr, edgecolor='white')
for b, v in zip(bars, cat_stats['IQR']):
    ax.text(v + 0.5, b.get_y() + b.get_height()/2,
            f'${v:.2f}', va='center', fontsize=9, color=TITLE_COLOR)
ax.set_title('Price IQR by Category (wider = more price variation)\n🔴 = Nest-USA',
             color=LABEL_COLOR, fontweight='bold')
ax.set_xlabel('IQR of Avg Price ($)', color=LABEL_COLOR)
ax.set_ylabel('Product Category', color=LABEL_COLOR)
ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.show()
print('Observation: Nest-USA has the widest price range ($0-$350) and highest IQR,')
print('indicating a diverse product lineup from budget to premium.')
print('Apparel, Office, and Drinkware are concentrated under $50 (low IQR).')

# ============================================================
# PROPOSAL PLOT 4: Monthly Revenue Heatmap by Category
# (프로포잘: 8-10월 계절성 피크, Nest-USA 매출 1위)
# ============================================================
print('\n--- Proposal Plot 4: Monthly Revenue Heatmap (Seasonality Analysis) ---')
fig, axes = plt.subplots(2, 1, figsize=(16, 14))
fig.suptitle('Seasonal Revenue Analysis by Product Category',
             color=TITLE_COLOR, fontweight='bold', fontsize=16)

# Top: Heatmap
ax = axes[0]
top8_cats = df['Product_Category'].value_counts().head(8).index.tolist()
monthly_heat = (df[df['Product_Category'].isin(top8_cats)]
                .groupby(['Product_Category', 'Month'])['Revenue']
                .sum()
                .unstack(fill_value=0))
# Normalize per row for pattern visibility
monthly_heat_norm = monthly_heat.div(monthly_heat.max(axis=1), axis=0).round(2)
sns.heatmap(monthly_heat_norm, annot=True, fmt='.2f', cmap='YlOrRd',
            ax=ax, linewidths=0.5, cbar_kws={'label': 'Normalized Revenue (0-1)'},
            annot_kws={'fontsize': 9})
ax.set_title('Normalized Monthly Revenue Heatmap (Top 8 Categories)',
             color=LABEL_COLOR, fontweight='bold')
ax.set_xlabel('Month', color=LABEL_COLOR)
ax.set_ylabel('Product Category', color=LABEL_COLOR)
# Highlight months 8-10
for month_idx in [7, 8, 9]:  # 0-indexed
    ax.add_patch(plt.Rectangle((month_idx, 0), 1, len(top8_cats),
                                fill=False, edgecolor='#2c3e50',
                                linewidth=2.5, linestyle='--'))
ax.text(8.5, -0.5, 'Peak\nSeason', ha='center', fontsize=10,
        color='#2c3e50', fontweight='bold')

# Bottom: Line plot for top 4 categories
ax = axes[1]
top4_cats = df['Product_Category'].value_counts().head(4).index.tolist()
monthly_line = (df[df['Product_Category'].isin(top4_cats)]
                .groupby(['Month', 'Product_Category'])['Revenue']
                .sum()
                .unstack(fill_value=0))
colors_line = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6']
markers = ['o', 's', '^', 'D']
for cat, col, mk in zip(top4_cats, colors_line, markers):
    ax.plot(monthly_line.index, monthly_line[cat],
            marker=mk, color=col, linewidth=2.5, markersize=8, label=cat)
    # annotate peak month
    peak_m = monthly_line[cat].idxmax()
    peak_v = monthly_line[cat].max()
    ax.annotate(f'Peak\nM{peak_m}',
                xy=(peak_m, peak_v),
                xytext=(peak_m + 0.3, peak_v * 1.02),
                fontsize=8, color=col, fontweight='bold')

# Shade months 8-10
ax.axvspan(7.5, 10.5, alpha=0.1, color='orange', label='Seasonal Peak (Aug-Oct)')
ax.set_title('Monthly Revenue Trend — Top 4 Categories',
             color=LABEL_COLOR, fontweight='bold')
ax.set_xlabel('Month', color=LABEL_COLOR)
ax.set_ylabel('Total Revenue ($)', color=LABEL_COLOR)
ax.set_xticks(range(1, 13))
ax.legend(title='Category', bbox_to_anchor=(1.01, 1), loc='upper left')
ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()
print('Observation: Clear seasonal peaks visible in months 8-10 across most categories.')
print('Nest-USA consistently generates the highest monthly revenue.')
print('The heatmap reveals that peak timing slightly varies by category,')
print('suggesting different marketing strategies may be needed per product line.')

print('\n' + '=' * 60)
print('ALL PROPOSAL INSIGHT PLOTS COMPLETE')
print('Phase I fully covers all proposal findings!')
print('=' * 60)

df.dropna(subset=['Online_Spend', 'Offline_Spend', 'Avg_Price', 'Tenure_Months', 'Quantity'], inplace=True)
num_cols = ['Tenure_Months', 'Quantity', 'Avg_Price', 'Delivery_Charges', 'GST', 'Offline_Spend', 'Online_Spend', 'Discount_pct']

# ============================================================
# PLOT 28: HEXBIN PLOT — Online vs Offline Spend density
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Spending Patterns Analysis', fontsize=16, fontweight='bold')

ax = axes[0, 0]
hb = ax.hexbin(df['Offline_Spend'], df['Online_Spend'], gridsize=30, cmap='YlOrRd', mincnt=1)
fig.colorbar(hb, ax=ax, label='Count')
ax.set_xlabel('Offline Spend')
ax.set_ylabel('Online Spend')
ax.set_title('Online vs Offline Spend Density')

ax = axes[0, 1]
hb2 = ax.hexbin(df['Avg_Price'], df['Online_Spend'], gridsize=30, cmap='Blues', mincnt=1)
fig.colorbar(hb2, ax=ax, label='Count')
ax.set_xlabel('Avg Price')
ax.set_ylabel('Online Spend')
ax.set_title('Avg Price vs Online Spend')

ax = axes[1, 0]
hb3 = ax.hexbin(df['Tenure_Months'], df['Online_Spend'], gridsize=25, cmap='Greens', mincnt=1)
fig.colorbar(hb3, ax=ax, label='Count')
ax.set_xlabel('Tenure Months')
ax.set_ylabel('Online Spend')
ax.set_title('Tenure vs Online Spend')

ax = axes[1, 1]
hb4 = ax.hexbin(df['Quantity'], df['Online_Spend'], gridsize=25, cmap='Purples', mincnt=1)
fig.colorbar(hb4, ax=ax, label='Count')
ax.set_xlabel('Quantity')
ax.set_ylabel('Online Spend')
ax.set_title('Quantity vs Online Spend')

plt.tight_layout()
plt.savefig('plot_28_hexbin.png', dpi=150, bbox_inches='tight')
plt.close()
print("PLOT 28 saved")

# ============================================================
# PLOT 29: STRIP PLOT
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Strip Plot Analysis', fontsize=16, fontweight='bold')

sns.stripplot(data=df, x='Gender', y='Online_Spend', hue='Coupon_Status',
              dodge=True, alpha=0.4, jitter=True, ax=axes[0], palette='Set2')
axes[0].set_title('Online Spend by Gender & Coupon Status')
axes[0].set_xlabel('Gender')
axes[0].set_ylabel('Online Spend')

sns.stripplot(data=df, x='Location', y='Online_Spend', hue='Gender',
              dodge=True, alpha=0.4, jitter=True, ax=axes[1], palette='Set1')
axes[1].set_title('Online Spend by Location & Gender')
axes[1].set_xlabel('Location')
axes[1].set_ylabel('Online Spend')
axes[1].tick_params(axis='x', rotation=30)

plt.tight_layout()
plt.savefig('plot_29_strip_plot.png', dpi=150, bbox_inches='tight')
plt.close()
print("PLOT 29 saved")

# ============================================================
# PLOT 30: SWARM PLOT (작은 샘플 사용 — 전체 데이터면 너무 느림)
# ============================================================
df_sample = df.sample(n=min(800, len(df)), random_state=42)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Swarm Plot Analysis', fontsize=16, fontweight='bold')

sns.swarmplot(data=df_sample, x='Gender', y='Online_Spend', hue='Coupon_Status',
              dodge=True, size=3, ax=axes[0], palette='Set2')
axes[0].set_title('Online Spend Distribution by Gender')
axes[0].set_xlabel('Gender')
axes[0].set_ylabel('Online Spend')

sns.swarmplot(data=df_sample, x='Coupon_Status', y='Avg_Price', hue='Gender',
              dodge=True, size=3, ax=axes[1], palette='Set1')
axes[1].set_title('Avg Price by Coupon Status')
axes[1].set_xlabel('Coupon Status')
axes[1].set_ylabel('Avg Price')

plt.tight_layout()
plt.savefig('plot_30_swarm_plot.png', dpi=150, bbox_inches='tight')
plt.close()
print("PLOT 30 saved")

# ============================================================
# PLOT 31: KDE MULTIVARIATE (subplot 방식)
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('KDE & Statistical Distribution Analysis', fontsize=16, fontweight='bold')

# KDE: Online Spend by Gender
for gender, grp in df.groupby('Gender'):
    grp['Online_Spend'].plot.kde(ax=axes[0,0], label=gender, linewidth=2)
axes[0,0].set_title('KDE: Online Spend by Gender')
axes[0,0].set_xlabel('Online Spend')
axes[0,0].legend()

# KDE: Avg Price by Coupon Status
for status, grp in df.groupby('Coupon_Status'):
    grp['Avg_Price'].plot.kde(ax=axes[0,1], label=status, linewidth=2)
axes[0,1].set_title('KDE: Avg Price by Coupon Status')
axes[0,1].set_xlabel('Avg Price')
axes[0,1].legend()

# KDE: Offline vs Online Spend overlay
df['Online_Spend'].plot.kde(ax=axes[1,0], label='Online', linewidth=2, color='steelblue')
df['Offline_Spend'].plot.kde(ax=axes[1,0], label='Offline', linewidth=2, color='coral')
axes[1,0].set_title('KDE: Online vs Offline Spend')
axes[1,0].set_xlabel('Spend Amount')
axes[1,0].legend()

# KDE: Tenure by Location (top 3)
top3_loc = df['Location'].value_counts().index[:3]
for loc in top3_loc:
    df[df['Location']==loc]['Tenure_Months'].plot.kde(ax=axes[1,1], label=loc, linewidth=2)
axes[1,1].set_title('KDE: Tenure by Location (Top 3)')
axes[1,1].set_xlabel('Tenure Months')
axes[1,1].legend()

plt.tight_layout()
plt.savefig('plot_31_stats_kde_multivariate.png', dpi=150, bbox_inches='tight')
plt.close()
print("PLOT 31 saved")

# ============================================================
# PLOT 32: OUTLIER DETECTION — Boxplot + IQR
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Outlier Detection Analysis', fontsize=16, fontweight='bold')

cols_to_check = ['Online_Spend', 'Offline_Spend', 'Avg_Price', 'Delivery_Charges']
colors = ['steelblue', 'coral', 'seagreen', 'mediumpurple']

for i, (col, color) in enumerate(zip(cols_to_check, colors)):
    ax = axes[i//2, i%2]
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower) | (df[col] > upper)][col]

    ax.boxplot(df[col].dropna(), vert=True, patch_artist=True,
               boxprops=dict(facecolor=color, alpha=0.6),
               medianprops=dict(color='black', linewidth=2))
    ax.set_title(f'{col}\n(Outliers: {len(outliers)}, {len(outliers)/len(df)*100:.1f}%)')
    ax.set_ylabel(col)
    ax.set_xticks([])

plt.tight_layout()
plt.savefig('plot_32_outlier_boxplot.png', dpi=150, bbox_inches='tight')
plt.close()
print("PLOT 32 saved")