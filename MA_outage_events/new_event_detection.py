import pandas as pd
import numpy as np

df = pd.read_csv('county_outage_blocks.csv')
df['start_time'] = pd.to_datetime(df['start_time'])
df['end_time'] = pd.to_datetime(df['end_time'])

three_hours = pd.Timedelta(hours=3)
all_events = []
event_id_counter = 1

# Process each county separately
for county in df['county'].unique():
    county_data = df[df['county'] == county].sort_values(by="start_time")
    current_event = None
    
    for _, row in county_data.iterrows():
        if current_event is None:
            current_event = row.copy()
            current_event_rows = [row]
        else:
            if row['start_time'] - current_event['end_time'] <= three_hours:
                # Extend event time
                current_event['end_time'] = max(current_event['end_time'], row['end_time'])
                # Update max customers_out
                current_event['customers_out'] = max(current_event['customers_out'], row['customers_out'])
                # Keep track of all rows in this event
                current_event_rows.append(row)
            else:
                # Calculate event metrics
                times = [r['start_time'] for r in current_event_rows] + [current_event_rows[-1]['end_time']]
                values = [r['customers_out'] for r in current_event_rows] + [current_event_rows[-1]['customers_out']]
                times_num = pd.Series(times).astype('int64') / 1e9  # seconds
                auc = np.trapz(values, times_num)  # AUC
                
                all_events.append({
                    'event_id': event_id_counter,
                    'county': county,
                    'latitude': np.mean([r['latitude'] for r in current_event_rows]),
                    'longitude': np.mean([r['longitude'] for r in current_event_rows]),
                    'event_start_time': current_event['start_time'],
                    'event_end_time': current_event['end_time'],
                    'area_under_curve': auc,
                    'no_of_customers_out': current_event['customers_out']
                })
                event_id_counter += 1
                current_event = row.copy()
                current_event_rows = [row]
    
    # Add last event for this county
    if current_event is not None and len(current_event_rows) > 0:
        times = [r['start_time'] for r in current_event_rows] + [current_event_rows[-1]['end_time']]
        values = [r['customers_out'] for r in current_event_rows] + [current_event_rows[-1]['customers_out']]
        times_num = pd.Series(times).astype('int64') / 1e9  # seconds
        auc = np.trapz(values, times_num)
        
        all_events.append({
            'event_id': event_id_counter,
            'county': county,
            'latitude': np.mean([r['latitude'] for r in current_event_rows]),
            'longitude': np.mean([r['longitude'] for r in current_event_rows]),
            'event_start_time': current_event['start_time'],
            'event_end_time': current_event['end_time'],
            'area_under_curve': auc,
            'no_of_customers_out': current_event['customers_out']
        })
        event_id_counter += 1

# Create final event dataset
event_dataset = pd.DataFrame(all_events)

# Save if needed
event_dataset.to_csv('county_event_dataset.csv', index=False)

print(event_dataset.head())
