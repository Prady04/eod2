from trendspy import Trends
import pandas as pd



if __name__ == "__main__":
    tr = Trends(request_delay=3.0)
    df = tr.interest_over_time(['python', 'javascript'])
    print(df)
    # Analyze geographic distribution
    #geo_df = tr.interest_by_region('python')

    # Get related queries
    related = tr.related_queries('python')
    print(related)
    categories = tr.categories(find='technology')
    # Output: [{'name': 'Computers & Electronics', 'id': '13'}, ...]
    # Search for locations
    locations = tr.geo(find='york')
    # Output: [{'name': 'New York', 'id': 'US-NY'}, ...]
    # Use in queries
    df = tr.interest_over_time(
        'python',
        geo='US-NY',      # Found location ID
        cat='13'          # Found category ID
    )
