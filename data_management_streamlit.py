import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
from streamlit_folium import st_folium
import folium
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
from wordcloud import WordCloud
from bs4 import BeautifulSoup
import requests
import re
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from unidecode import unidecode

# Page configuration
st.set_page_config(
    page_title="Vehicle Market Analysis",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for improved design
st.markdown("""
    <style>
    /* Header Styling */
    .purple-header {
        color: #ffffff; /* White text for contrast */
        background: linear-gradient(90deg, #e6ccff 0%, #d9b3ff 100%); /* Soft light purple gradient */
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 30px;
        font-size: 28px;
        font-weight: bold;
        box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.2); /* Subtle shadow for pop */
        letter-spacing: 1px; /* Slight letter spacing for readability */
    }

    /* Information Box Styling */
    .info-box {
        background-color: #f3e8fd; /* Light purple */
        padding: 20px;
        border-radius: 15px;
        margin: 15px 0;
        font-size: 16px;
        color: #333;
        border: 1px solid #e1d6f5; /* Subtle border */
    }

    /* Findings Box Styling */
    .findings-box {
        background-color: #fff9e6; /* Light yellow */
        padding: 15px;
        border-radius: 15px;
        margin: 15px 0;
        font-size: 16px;
        font-style: italic;
        border-left: 5px solid #ffd700; /* Golden accent */
    }

    /* Team Box Styling */
    .team-box {
        background-color: #ececec; /* Light gray */
        padding: 20px;
        border-radius: 15px;
        margin: 15px 0;
        font-size: 18px;
        color: #555;
        border: 1px solid #ccc; /* Subtle border */
        box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1);
    }

    /* Metrics Box Styling */
    .metric-box {
        background-color: #e0f7fa; /* Light teal */
        padding: 15px;
        border-radius: 15px;
        text-align: center;
        font-size: 20px;
        font-weight: bold;
        color: #00796b; /* Deep teal */
        margin-bottom: 10px;
    }

    /* Add spacing and consistent margins */
    .section {
        margin-bottom: 40px;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_data
def load_and_preprocess_data():
    """Load and preprocess the dataset"""
    data_path = "/Users/ecembayindir/Desktop/SORBONNE/Classes/Data Management, Dataviz, Text Mining/final_project_data_management/final_version/vehicles.csv"
    df = pd.read_csv(data_path)

    # Convert posting_date to datetime without specifying format
    df['posting_date'] = pd.to_datetime(df['posting_date'], errors='coerce', utc=True)

    # Handle missing values
    df.fillna({
        'year': df['year'].median(),
        'odometer': df['odometer'].mean(),
        'manufacturer': 'Unknown',
        'model': 'Unknown',
        'fuel': 'Unknown',
        'title_status': 'Unknown',
        'transmission': 'Unknown',
        'type': 'Unknown',
        'paint_color': 'Unknown',
        'condition': 'Unknown',
        'cylinders': 'Unknown',
        'drive': 'Unknown',
        'VIN': 'Not Specified',
    }, inplace=True)

    return df


def create_derived_variables(df):
    """Create derived variables"""

    def categorize_price(price):
        if price < 5000:
            return 'Budget'
        elif 5000 <= price < 15000:
            return 'Mid-range'
        elif 15000 <= price < 30000:
            return 'Top of the range'
        else:
            return 'Premium'

    df['price_category'] = df['price'].apply(categorize_price)
    df['vehicle_age'] = 2024 - df['year']
    df.loc[df['vehicle_age'] < 0, 'vehicle_age'] = np.nan

    df['km_per_year'] = df['odometer'] / (df['vehicle_age'].replace(0, np.nan))
    df['wear_category'] = pd.cut(
        df['km_per_year'],
        bins=[0, 10000, 20000, 30000, np.inf],
        labels=['Very good condition', 'Good condition', 'Average condition', 'Worn out']
    )

    price_map = {'Budget': 1, 'Mid-range': 2, 'Top of the range': 3, 'Premium': 4}
    df['value_score'] = df['price_category'].map(price_map) / (df['vehicle_age'] + 1)

    df['vehicle_segment'] = df['manufacturer'] + ' - ' + df['type']

    df['mileage_category'] = pd.cut(
        df['odometer'],
        bins=[0, 30000, 70000, float('inf')],
        labels=['Low', 'Medium', 'High']
    )

    return df


def display_findings(findings):
    """Display findings in a formatted box"""
    st.markdown("""
        <div class="findings-box">
            <h3 style="color:purple; font-family:Verdana; font-weight:bold;">🔎 Findings:</h3>
            <ul style="color:#333; font-family:Arial; line-height:1.6;">
    """, unsafe_allow_html=True)

    for finding in findings:
        st.markdown(f"<li><strong>{finding}</strong></li>", unsafe_allow_html=True)

    st.markdown("</ul></div>", unsafe_allow_html=True)


def create_manufacturer_analysis(df):
    """Create manufacturer analysis visualizations"""
    st.subheader("Manufacturer Analysis")

    # Most popular manufacturers
    manufacturer_counts = df['manufacturer'].value_counts().head(10)
    fig1 = px.bar(
        manufacturer_counts,
        title="Top 10 Car Manufacturers by Popularity",
        color_discrete_sequence=['skyblue'],
        labels={'index': 'Manufacturer', 'value': 'Number of Cars'}
    )
    st.plotly_chart(fig1, use_container_width=True)

    display_findings([
        "Ford appears to be the most popular car manufacturer",
        "Chevrolet and Toyota follow as the next most popular brands",
        "American and Japanese manufacturers dominate the top spots"
    ])

    # Manufacturer popularity by year
    df_year = df.groupby(['manufacturer', 'year']).size().reset_index(name='count')
    df_year = df_year[(df_year['year'] > 2004) & (df_year['count'] > 100)]

    fig2 = px.bar(
        df_year,
        x='manufacturer',
        y='count',
        animation_frame='year',
        title='Car Manufacturers Popularity by Year',
        color='manufacturer'
    )
    st.plotly_chart(fig2, use_container_width=True)

    display_findings([
        "Ford and Chevrolet maintain leadership across years",
        "Market share distribution remains relatively stable",
        "Recent years show increasing diversity in manufacturer popularity"
    ])


def create_price_analysis(df):
    """Create price analysis visualizations"""
    st.subheader("Price Analysis")

    col1, col2 = st.columns(2)

    with col1:
        # Average price by manufacturer
        avg_price = df.groupby('manufacturer')['price'].mean().sort_values(ascending=False).head(10)
        fig1 = px.bar(
            avg_price,
            title="Top 10 Manufacturers by Average Car Price",
            color_discrete_sequence=['orange'],
            labels={'index': 'Manufacturer', 'value': 'Average Price ($)'}
        )
        st.plotly_chart(fig1, use_container_width=True)

        display_findings([
            "Premium brands like Volvo and Mercedes have the highest average prices",
            "Luxury brands command significantly higher prices",
            "Clear price differentiation between luxury and mainstream manufacturers"
        ])

    with col2:
        # Price category distribution
        price_dist = df['price_category'].value_counts()
        fig2 = px.bar(
            price_dist,
            title="Distribution of Vehicle Price Categories",
            color=price_dist.index,
            color_discrete_sequence=['#8A2BE2', '#9B30FF', '#7A378B', '#4B0082'],
            labels={'index': 'Price Category', 'value': 'Number of Vehicles'}
        )
        st.plotly_chart(fig2, use_container_width=True)

        display_findings([
            "Mid-range cars have more cars listed",
            "Premium segment represents a smaller portion of the market",
            "Budget and top-range categories show balanced distribution"
        ])

    # Price category by manufacturer
    st.subheader("Price Categories by Manufacturer")
    pivot = pd.crosstab(df['manufacturer'], df['price_category'])
    fig3 = px.bar(
        pivot,
        title="Distribution of Price Categories by Manufacturer",
        barmode="stack",
        labels={'value': 'Number of Vehicles', 'manufacturer': 'Manufacturer'}
    )
    fig3.update_layout(xaxis={'tickangle': 45})
    st.plotly_chart(fig3, use_container_width=True)

    display_findings([
        "Ford and Chevrolet dominate across all price categories",
        "Honda, Nissan, and Toyota focus mainly on mid-range vehicles",
        "Luxury brands concentrate in premium category"
    ])


def create_geographical_analysis(df):
    """Create geographical analysis visualizations"""
    st.subheader("Geographical Analysis")

    # State distribution - Horizontal bar chart
    state_counts = df.groupby('state').size().reset_index(name='count')
    fig1 = px.bar(
        state_counts,
        x='count',
        y='state',
        orientation='h',
        title='Number of Cars by State',
        labels={'count': 'Number of Cars', 'state': 'State'},
        color='count',
        color_continuous_scale='Blues'
    )
    fig1.update_layout(
        xaxis_title="Number of Cars",
        yaxis_title="State",
        yaxis=dict(categoryorder='total ascending'),
        width=800,
        height=600,
        title={
            'text': 'Number of Cars by State',
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        }
    )
    st.plotly_chart(fig1, use_container_width=True)

    display_findings([
        "California has the highest number of cars listed, with over 51,000 vehicles."
    ])

    # Vehicle types by region - Heatmap
    type_region = pd.crosstab(df['type'], df['region'])
    top_regions = type_region.sum().nlargest(10).index
    type_region_filtered = type_region[top_regions]
    fig2 = px.imshow(
        type_region_filtered,
        title="Distribution of Car Types Across Regions",
        aspect="auto",
        color_continuous_scale="YlGnBu"
    )
    st.plotly_chart(fig2, use_container_width=True)

    display_findings([
        "SUVs and Sedans dominate across all regions.",
        "Pickup trucks are more common in rural regions.",
        "Urban areas show higher diversity in vehicle types."
    ])

    # Dominant paint colors by region - Scatter Mapbox
    df_sampled = df.sample(n=5000, random_state=42)
    color_counts = df_sampled.groupby(['region', 'paint_color']).size().reset_index(name='count')
    color_counts = color_counts.sort_values(by=['region', 'count'], ascending=[True, False])
    dominant_color_map = color_counts.groupby('region').head(3).reset_index()
    dominant_color_map = dominant_color_map.merge(
        df_sampled[['region', 'lat', 'long']].drop_duplicates(), on='region', how='left'
    )
    color_mapping = {
        "Unknown": "lightgrey", "white": "pink", "brown": "brown", "black": "black",
        "red": "red", "silver": "silver", "blue": "blue", "custom": "gold",
        "yellow": "yellow", "grey": "grey", "purple": "purple", "green": "green", "orange": "orange"
    }
    fig3 = px.scatter_mapbox(
        dominant_color_map,
        lat='lat',
        lon='long',
        hover_name='region',
        hover_data=['paint_color', 'count'],
        color='paint_color',
        color_discrete_map=color_mapping,
        zoom=4,
        title='Dominant Paint Colors by Region (Sampled)',
        opacity=0.5
    )
    fig3.update_layout(mapbox_style='open-street-map', margin={'r': 0, 't': 30, 'l': 0, 'b': 0})
    st.plotly_chart(fig3, use_container_width=True)

    display_findings([
        "Identifies regional preferences for vehicle colors.",
        "Highlights unique color preferences in certain areas, such as bright colors for specific vehicle types."
    ])

    # Filter out rows with NaN values in lat and long
    df_filtered = df.dropna(subset=['lat', 'long'])

    st.markdown(
        '<h4 style="text-align:left; font-family:Verdana; font-size:15px; color:#31333F; margin-bottom:20px;">'
        'Geographical Distribution of Vehicle According to Prices</h4>',
        unsafe_allow_html=True
    )

    # Geographical distribution of vehicle listings by price - Folium Map
    m = folium.Map(
        location=[39.8283, -98.5795],
        zoom_start=4,
        tiles="cartodbpositron",
        width=800,
        height=500
    )

    # Add markers to the map
    for _, row in df_filtered.sample(1000).iterrows():  # Limit to 1000 samples for performance
        folium.CircleMarker(
            location=[row['lat'], row['long']],
            radius=min(row['price'] / 10000, 10),  # Scale radius by price
            color='blue',
            fill=True,
            fill_color='blue',
            fill_opacity=0.3,
            popup=folium.Popup(
                f"<b>Price:</b> ${row['price']}<br>"
                f"<b>Manufacturer:</b> {row['manufacturer']}<br>"
                f"<b>Type:</b> {row['type']}<br>"
                f"<b>Fuel:</b> {row['fuel']}<br>"
                f"<b>Odometer:</b> {row['odometer']} miles",
                max_width=300
            )
        ).add_to(m)

    # Use streamlit-folium to render the map in Streamlit
    st_folium(m, width=800, height=500)

    display_findings([
        "Higher prices cluster around urban and coastal areas."
    ])

    # Average Price vs Mileage for Top 10 Regions
    region_stats = df.groupby('region').agg({
        'price': 'mean',
        'odometer': 'mean'
    }).reset_index()
    region_stats.rename(columns={'price': 'avg_price', 'odometer': 'avg_mileage'}, inplace=True)

    # Get the top 10 regions by average price
    top_regions = region_stats.nlargest(10, 'avg_price')

    # Create a scatter plot using Plotly
    fig5 = px.scatter(
        top_regions,
        x='avg_mileage',
        y='avg_price',
        color='region',
        size='avg_price',
        title='Average Price vs Mileage for Top 10 Regions by Price',
        labels={
            'avg_mileage': 'Average Mileage',
            'avg_price': 'Average Price ($)'
        },
        hover_name='region'
    )
    fig5.update_layout(
        xaxis_title='Average Mileage',
        yaxis_title='Average Price',
        legend_title='Region',
        width=800,
        height=600
    )
    st.plotly_chart(fig5, use_container_width=True)

    display_findings([
        "Regions like Delaware, Frederick, and Humboldt County have high average prices with lower average mileage.",
        "Indicates demand for premium vehicles in specific areas."
    ])

def create_correlation_analysis(df):
    """Create correlation heatmap analysis"""
    st.subheader("Correlation Analysis")

    # Select columns for correlation analysis
    correlation_columns = ['price', 'value_score', 'odometer', 'vehicle_age', 'km_per_year']

    # Check if all required columns are in the dataframe
    missing_columns = [col for col in correlation_columns if col not in df.columns]
    if missing_columns:
        st.error(f"Missing columns for correlation analysis: {', '.join(missing_columns)}")
        return

    # Compute the correlation matrix
    correlation_matrix = df[correlation_columns].corr()

    # Display the correlation heatmap using seaborn
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='magma', center=0, ax=ax)
    ax.set_title('Correlation Matrix')
    st.pyplot(fig)

    # Display findings
    display_findings([
        "Value Score has a strong negative correlation (-0.57) with Vehicle Age, indicating that older vehicles tend to have lower value scores.",
        "Kilometers per Year and Odometer show a moderate positive correlation (0.59), suggesting that higher annual mileage contributes to higher overall odometer readings."
    ])


def create_vehicle_characteristics(df):
    """Create vehicle characteristics visualizations"""
    st.subheader("Vehicle Characteristics")

    # Plot 1: Histogram of Vehicle Age with KDE
    fig1 = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Distribution of Vehicle Age", "Boxplot of Vehicle Age")
    )

    # Histogram of Vehicle Age
    hist_data = df['vehicle_age']
    kde = sns.kdeplot(hist_data, color='blue', lw=2)
    kde_y = kde.get_lines()[0].get_ydata()
    kde_x = kde.get_lines()[0].get_xdata()

    fig1.add_trace(
        go.Histogram(
            x=hist_data,
            xbins=dict(size=5),  # Bin size of 5 years
            marker=dict(color='blue', line=dict(color='black', width=1)),
            opacity=0.75,
            name="Histogram"
        ),
        row=1, col=1
    )

    # Add KDE line
    fig1.add_trace(
        go.Scatter(
            x=kde_x,
            y=kde_y * len(hist_data) * 5,  # Scale KDE to match histogram
            mode='lines',
            line=dict(color='blue', width=2),
            name="KDE Curve"
        ),
        row=1, col=1
    )

    # Boxplot of Vehicle Age
    fig1.add_trace(
        go.Box(
            y=hist_data,
            marker=dict(color='blue'),
            name="Boxplot"
        ),
        row=1, col=2
    )

    fig1.update_layout(
        showlegend=False,  # Hide legend
        height=450,
        width=1000,
        margin=dict(t=40, b=40, l=40, r=40),
        font=dict(size=12),
    )
    fig1.update_annotations(font=dict(size=14, color="purple"))
    fig1.update_xaxes(title_text="Vehicle Age (years)", row=1, col=1)
    fig1.update_yaxes(title_text="Number of Vehicles", row=1, col=1)
    fig1.update_yaxes(title_text="Vehicle Age (years)", row=1, col=2)

    # Display the first visualization
    st.plotly_chart(fig1, use_container_width=True)

    # Plot 2: Mileage per Year
    col1, col2 = st.columns(2)

    with col1:
        df_filtered = df[df['km_per_year'] < 100000]
        fig2 = px.histogram(
            df_filtered,
            x='km_per_year',
            title="Distribution of Mileage per Year",
            color_discrete_sequence=['green'],
            nbins=20
        )
        fig2.update_layout(
            xaxis_title="Mileage per Year",
            yaxis_title="Number of Vehicles",
            title_x=0.5,
            margin=dict(t=40, b=40, l=40, r=40),
            font=dict(size=12)
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Plot 3: Wear Categories
    with col2:
        wear_counts = df['wear_category'].value_counts().reset_index()
        wear_counts.columns = ['wear_category', 'count']

        fig3 = px.bar(
            wear_counts,
            x='wear_category',
            y='count',
            title="Distribution of Vehicle Wear Categories",
            color='wear_category',
            color_discrete_sequence=['#0066cc', '#0099ff', '#00ccff', '#33ccff']
        )
        fig3.update_layout(
            xaxis_title="Vehicle Condition",
            yaxis_title="Number of Vehicles",
            title_x=0.5,
            margin=dict(t=40, b=40, l=40, r=40),
            font=dict(size=12)
        )
        st.plotly_chart(fig3, use_container_width=True)

    # Display Findings
    display_findings([
        "The majority of vehicles are less than 20 years old.",
        "The histogram shows a steep decline for older vehicles.",
        "Boxplot highlights significant outliers for vehicles older than 50 years.",
        "Most vehicles show annual mileage under 20,000 km.",
        "Very good condition vehicles dominate the listings."
    ])


def create_fuel_transmission_analysis(df):
    """Create fuel and transmission analysis visualizations in Streamlit."""
    st.subheader("Fuel and Transmission Analysis")

    # Fuel Types Distribution
    col1, col2 = st.columns(2)

    with col1:
        # Correct grouping for fuel types
        by_fuel = df.groupby('fuel').size().reset_index(name='count')
        by_fuel['ratio'] = round(by_fuel['count'] / by_fuel['count'].sum() * 100, 2)

        # Create pie chart for fuel types
        fig_fuel = go.Figure(
            data=[
                go.Pie(
                    labels=by_fuel['fuel'],
                    values=by_fuel['ratio'],
                    marker=dict(colors=px.colors.qualitative.Pastel, line=dict(color='#FFFFFF', width=2))
                )
            ]
        )
        fig_fuel.update_layout(
            title={
                'text': 'Car Fuel Types Distribution',
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            },
            width=700,
            height=600
        )
        st.plotly_chart(fig_fuel, use_container_width=True)

    with col2:
        # Correct grouping for transmission types
        by_transmission = df.groupby('transmission').size().reset_index(name='count')
        by_transmission['ratio'] = round(by_transmission['count'] / by_transmission['count'].sum() * 100, 2)

        # Create pie chart for transmission types
        fig_transmission = go.Figure(
            data=[
                go.Pie(
                    labels=by_transmission['transmission'],
                    values=by_transmission['ratio'],
                    marker=dict(colors=px.colors.qualitative.Pastel, line=dict(color='#FFFFFF', width=2))
                )
            ]
        )
        fig_transmission.update_layout(
            title={
                'text': 'Car Transmission Types',
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            },
            width=700,
            height=600
        )
        st.plotly_chart(fig_transmission, use_container_width=True)

    # Average Price by Title Status
    avg_price_title = df.groupby('title_status')['price'].mean().reset_index(name='average_price')
    fig_price_title = px.bar(
        avg_price_title,
        x='title_status',
        y='average_price',
        title='Average Price by Car Title Status',
        labels={'average_price': 'Average Price ($)', 'title_status': 'Car Title Status'},
        color='average_price',
        color_continuous_scale='Viridis'
    )
    fig_price_title.update_layout(
        width=800,
        height=600,
        title={
            'text': 'Average Price by Car Title Status',
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        }
    )
    st.plotly_chart(fig_price_title, use_container_width=True)

    # Findings
    display_findings([
        "Gas is the most common fuel type, accounting for the majority of listings.",
        "Automatic transmissions dominate the market.",
        "Clean title vehicles command higher average prices."
    ])


def create_additional_insights(df):
    """Create additional insights visualizations"""
    st.subheader("Additional Insights")

    # Value score analysis
    fig2 = px.scatter(
        df.sample(1000),
        x='vehicle_age',
        y='value_score',
        title="Value Score vs Vehicle Age",
        trendline="lowess",
        color_discrete_sequence=['purple']
    )
    st.plotly_chart(fig2, use_container_width=True)

    display_findings([
        "Value scores decrease significantly as the vehicle age increases, indicating lower perceived value for older cars.",
        "High value scores are concentrated in cars less than 20 years old, reflecting their higher demand or better condition."
    ])

# Average listing duration by vehicle type
    df['vehicle_age'] = 2024 - df['year']  # Calculate vehicle age if not already present
    avg_listing_duration = df.groupby('type')['vehicle_age'].mean().sort_values()

    fig4, ax = plt.subplots(figsize=(10, 6))
    avg_listing_duration.plot(kind='barh', color='orange', ax=ax)
    ax.set_title('Average Listing Duration by Vehicle Type', fontsize=16)
    ax.set_xlabel('Average Age (Years)', fontsize=12)
    ax.set_ylabel('Vehicle Type', fontsize=12)
    st.pyplot(fig4)

    display_findings([
        "Offroads and buses tend to have longer listing durations compared to SUVs and sedans."
    ])


def create_word_cloud():
    """Generate and display a Word Cloud from a scraped article"""
    st.subheader("Word Cloud Analysis")

    # Step 1: Scrape the article content
    url = "https://lemagdelaconso.ouest-france.fr/article-46-pourquoi-voitures-occasion-coutent-de-plus-en-plus-cher.html"
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        paragraphs = soup.find_all('p')
        article_text = " ".join([p.get_text() for p in paragraphs])
    else:
        st.error(f"Failed to retrieve the article. Status code: {response.status_code}")
        return

    # Step 2: Preprocess the text
    # Full list of stopwords
    stopWords = ['a', 'abord', 'absolument', 'afin', 'ai', 'aie', 'ainsi', 'ait', 'allaient', 'allo', 'allons', 'alors',
                 'an', 'année', 'ans',
                 'après', 'as', 'assez', 'attendu', 'au', 'aucun', 'aucune', 'aujourd', 'aujourd\'hui', 'aupres',
                 'aura', 'auraient',
                 'aurait', 'auront', 'aussi', 'autre', 'autrefois', 'autrement', 'autres', 'aux', 'avaient', 'avais',
                 'avait', 'avant',
                 'avec', 'avoir', 'avons', 'ayant', 'bah', 'beaucoup', 'bien', 'bonjour', 'bref', 'c', 'ça', 'car',
                 'ce', 'ceci', 'cela',
                 'celle', 'celle-ci', 'celle-là', 'celles', 'celles-ci', 'celles-là', 'celui', 'celui-ci', 'celui-là',
                 'cent', 'cependant',
                 'certain', 'certaine', 'certaines', 'certains', 'certes', 'ces', 'cet', 'cette', 'ceux', 'ceux-ci',
                 'ceux-là', 'chacun',
                 'chacune', 'chaque', 'chez', 'ci', 'cinq', 'combien', 'comme', 'comment', 'compris', 'concernant',
                 'contre', 'coups',
                 'dans', 'de', 'debout', 'dedans', 'dehors', 'depuis', 'derrière', 'des', 'dès', 'désormais', 'deux',
                 'devant', 'devers',
                 'devra', 'doit', 'donc', 'dont', 'du', 'elle', 'elle-même', 'elles', 'elles-mêmes', 'en', 'encore',
                 'entre', 'envers',
                 'environ', 'es', 'est', 'et', 'étaient', 'étais', 'était', 'étant', 'etc', 'été', 'être', 'eu', 'eux',
                 'eux-mêmes', 'fais',
                 'faisaient', 'faisant', 'fait', 'façon', 'feront', 'font', 'gens', 'grâce', 'ha', 'hé', 'hein', 'hors',
                 'ici', 'il', 'ils',
                 'je', 'jusqu', 'jusque', 'là', 'la', 'laquelle', 'le', 'lequel', 'les', 'lesquelles', 'lesquels',
                 'leur', 'leurs', 'lui',
                 'lui-même', 'ma', 'maintenant', 'mais', 'me', 'même', 'mêmes', 'mes', 'mien', 'mienne', 'miennes',
                 'miens', 'moi', 'moi-même',
                 'moins', 'mon', 'même', 'n', 'ne', 'ni', 'nom', 'non', 'nos', 'notre', 'nous', 'nous-mêmes', 'nul',
                 'o', 'où', 'oh', 'oui',
                 'ou', 'ouais', 'par', 'parce', 'parfois', 'parle', 'parlent', 'parler', 'parmi', 'pas', 'pendant',
                 'peu', 'peut', 'peuvent',
                 'peux', 'plus', 'plutôt', 'pour', 'pourquoi', 'près', 'puis', 'puisque', 'qu', 'quand', 'que', 'quel',
                 'quelle', 'quelles',
                 'quels', 'qui', 'sa', 'sans', 'sauf', 'se', 'sera', 'seront', 'ses', 'seul', 'seule', 'seulement',
                 'si', 'sien', 'sienne',
                 'siennes', 'siens', 'sinon', 'soi', 'soi-même', 'soit', 'son', 'sont', 'sous', 'souvent', 'soyez',
                 'suis', 'sur', 't', 'ta',
                 'te', 'tes', 'toi', 'toi-même', 'ton', 'toujours', 'tous', 'tout', 'toute', 'toutes', 'tu', 'un',
                 'une', 'unes', 'uns', 'vos',
                 'votre', 'vous', 'vous-mêmes', 'vu', 'y', 'ça', 'étaient', 'état', 'étions', 'été', 'être']

    stopWords = [unidecode(sw) for sw in stopWords]
    stemmer = SnowballStemmer('french')

    def no_stop_word(string, stopWords):
        return ' '.join([word for word in string.split() if word not in stopWords])

    def stemmatise_text(text, stemmer):
        return ' '.join([stemmer.stem(word) for word in text.split()])

    def stem_cleaner(text, stemmer, stopWords):
        text = text.lower()
        text = unidecode(text)
        text = re.sub(r"{0-9}{4}", "annee", text)
        text = re.sub(r"[^a-z]+", " ", text)
        text = no_stop_word(text, stopWords)
        text = stemmatise_text(text, stemmer)
        return text

    cleaned_text = stem_cleaner(article_text, stemmer, stopWords)

    # Step 3: Generate Word Cloud
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white',
        colormap='viridis'
    ).generate(cleaned_text)

    # Step 4: Display the Word Cloud
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title('Word Cloud of Preprocessed Article Content', fontsize=16)
    st.pyplot(fig)

    # Step 5: Display Findings
    display_findings([
        "Common words like 'voitures', 'occasion', and 'prix' dominate the article.",
        "Strong focus on price increases and market factors."
    ])

def display_cleaned_data_info(df):
    """Display the final shape of the cleaned dataset"""
    st.subheader("Cleaned Dataset Information")

    # Display the dataset shape
    total_rows, total_columns = df.shape
    st.markdown(f"### Final Dataset Shape")

    # Add key metrics in two columns
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Records After Cleaning", "420,331")
    with col2:
        st.metric("Total Variables After Cleaning", "24")

    # Additional Insights
    st.write("The dataset has been cleaned and reduced to contain meaningful information.")
    st.write("This includes removing missing values and creating additional variables for more insightful analysis.")

def main():
    """Main function to run the Streamlit app"""
    st.markdown('<h1 class="purple-header">Vehicle Market Analysis</h1>', unsafe_allow_html=True)

    # Sidebar navigation
    st.sidebar.title("Navigation")
    section = st.sidebar.selectbox(
        "Choose Section",
        ["Dataset Overview",
         "Dataset Cleaning",
         "New Variable Creation",
         "Analysis Dashboard"]
    )

    # Load data
    df = load_and_preprocess_data()
    df_with_new_variables = create_derived_variables(df)

    if section == "Dataset Overview":
        # Project Team
        st.markdown("# 🚗 Vehicle Market Analysis", unsafe_allow_html=True)

        # Team Box
        st.markdown("""
            ### 📌 Project Participants
            * Ecem BAYINDIR
            * Kenny NUNGU
            * Traore ISSAKA MOUSSA
            * Juste BOTTHY
            """)

        # Dataset Overview
        st.markdown("## Dataset Overview: Craigslist Cars and Trucks")
        st.write("""
            This dataset provides detailed information about cars and trucks listed on Craigslist, 
            compiled from various regions across the United States. It offers valuable insights for 
            analyzing trends in vehicle sales, pricing, and other factors.
            """)

        # Dataset Summary
        st.markdown("### Dataset Summary")
        st.markdown("* **Number of Entries:** 426,880 rows")
        st.markdown("* **Number of Variables:** 26 columns")

        # Variables
        st.markdown("### Variables")
        variables = {
            "id": "Unique identifier for each listing",
            "url": "URL of the listing",
            "region": "Region where the listing is posted",
            "region_url": "URL of the Craigslist region",
            "price": "Vehicle price",
            "year": "Year of manufacture",
            "manufacturer": "Vehicle manufacturer",
            "model": "Vehicle model",
            "condition": "Condition of the vehicle",
            "cylinders": "Engine cylinder count",
            "fuel": "Fuel type (e.g., gasoline, diesel, electric)",
            "odometer": "Vehicle mileage",
            "title_status": "Status of the vehicle title",
            "transmission": "Transmission type (e.g., manual, automatic)",
            "VIN": "Vehicle Identification Number",
            "drive": "Drive type (e.g., 4wd, fwd, rwd)",
            "size": "Size category of the vehicle",
            "type": "Vehicle type (e.g., sedan, SUV, truck)",
            "paint_color": "Exterior paint color",
            "image_url": "URL of the listing image",
            "description": "Description of the listing",
            "county": "County information (not provided in this dataset)",
            "state": "State where the listing is located",
            "lat": "Latitude coordinates of the listing",
            "long": "Longitude coordinates of the listing",
            "posting_date": "Date the listing was posted"
        }

        # Create two columns for better layout
        col1, col2 = st.columns(2)

        # Split variables dictionary into two parts
        vars_list = list(variables.items())
        half = len(vars_list) // 2

        # Display variables in two columns
        with col1:
            for var, desc in vars_list[:half]:
                st.markdown(f"* **{var}:** {desc}")

        with col2:
            for var, desc in vars_list[half:]:
                st.markdown(f"* **{var}:** {desc}")

        # Show current dataset stats
        st.markdown("### Current Dataset Statistics")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Lines", f"{len(df):,}")
        with col2:
            st.metric("Unique Manufacturers", df['manufacturer'].nunique())
    elif section == "Dataset Cleaning":
        # Call the display_cleaned_data_info function for the new section
        display_cleaned_data_info(df_with_new_variables)
    elif section == "New Variable Creation":
        st.markdown("<h2>Creation of New Variables</h2>", unsafe_allow_html=True)
        st.code("""
def create_derived_variables(df):
    # 1. Price category
    def categorize_price(price):
        if price < 5000:
            return 'Budget'
        elif 5000 <= price < 15000:
            return 'Mid-range'
        elif 15000 <= price < 30000:
            return 'Top of the range'
        else:
            return 'Premium'

    df['price_category'] = df['price'].apply(categorize_price)

    # 2. Vehicle age
    df['vehicle_age'] = 2024 - df['year']

    # 3. Mileage/age ratio (vehicle wear)
    df['km_per_year'] = df['odometer'] / df['vehicle_age']
    df['wear_category'] = pd.cut(
        df['km_per_year'],
        bins=[0, 10000, 20000, 30000, np.inf],
        labels=['Very good condition', 'Good condition', 'Average condition', 'Worn out']
    )

    # 4. Value indicator
    price_map = {'Budget': 1, 'Mid-range': 2, 'Top of the range': 3, 'Premium': 4}
    df['value_score'] = df['price_category'].map(price_map) / (df['vehicle_age'] + 1)

    # 5. Mileage category
    df['mileage_category'] = pd.cut(
        df['odometer'],
        bins=[0, 30000, 70000, float('inf')],
        labels=['Low', 'Medium', 'High']
    )

    return df
        """)

    elif section == "Analysis Dashboard":
        analysis_type = st.sidebar.selectbox(
            "Choose Analysis Type",
            ["Manufacturer Analysis",
             "Price Analysis",
             "Correlation Analysis",
             "Geographical Analysis",
             "Vehicle Characteristics",
             "Fuel and Transmission",
             "Additional Insights",
             "Article Analysis"]
        )

        if analysis_type == "Manufacturer Analysis":
            create_manufacturer_analysis(df_with_new_variables)
        elif analysis_type == "Price Analysis":
            create_price_analysis(df_with_new_variables)
        elif analysis_type == "Correlation Analysis":
            create_correlation_analysis(df)
        elif analysis_type == "Geographical Analysis":
            create_geographical_analysis(df_with_new_variables)
        elif analysis_type == "Vehicle Characteristics":
            create_vehicle_characteristics(df_with_new_variables)
        elif analysis_type == "Fuel and Transmission":
            create_fuel_transmission_analysis(df_with_new_variables)
        elif analysis_type == "Additional Insights":
            create_additional_insights(df_with_new_variables)
        elif analysis_type == "Article Analysis":
            create_word_cloud()


if __name__ == "__main__":
    main()
