import numpy as np
import sklearn as sk
import matplotlib.pyplot as plt
#import pypots as pots
import random
import csv
import pandas as pd 


# Load the csv data

class Dataops:
    def __init__(self):
        self.raw_data = None
        self.validMoodWindow = []
        self.validAnxietyWindow = []
    
    def loadData(self):
        self.raw_data = pd.read_csv('raw_data_with_gender.csv', sep=',', usecols=['days_trend_mood0','days_trend_anxiety0'] ) # Load the csv file into a pandas DataFrame
        print(f"Loaded {len(self.raw_data)} rows of data")
        return self.raw_data

    def dataparse(self):
        self.validMoodWindow = []
        self.validAnxietyWindow = []
        self.smallMoodWindow = []
        self.smallAnxietyWindow = []
        self.invalidMoodEntries = []
        self.invalidAnxietyEntries = []

        # Parse mood data
        moodRows = self.raw_data.iloc[:, 0].values.tolist()
        for each in moodRows:
            if isinstance(each, str):
                stripped = each.strip().lstrip('[').rstrip(']')
                entry_count = len(stripped.split(','))
                if entry_count >= 4:
                    self.validMoodWindow.append(stripped)
                elif entry_count == 3:
                    self.smallMoodWindow.append(stripped)
                elif entry_count == 2:
                    self.invalidMoodEntries.append(stripped)
        
        print(f"Valid mood windows found (>=4 days): {len(self.validMoodWindow)}")
        print(f"Small mood windows found (3 days): {len(self.smallMoodWindow)}")
        print(f"Invalid mood windows found (2 days): {len(self.invalidMoodEntries)}")
                    
        # Parse anxiety data
        anxietyRows = self.raw_data.iloc[:, 1].values.tolist()
        for each in anxietyRows:
            if isinstance(each, str):
                stripped = each.strip().lstrip('[').rstrip(']')
                entry_count = len(stripped.split(','))
                if entry_count >= 4:
                    self.validAnxietyWindow.append(stripped)
                elif entry_count == 3:
                    self.smallAnxietyWindow.append(stripped)
                elif entry_count == 2:
                    self.invalidAnxietyEntries.append(stripped)
        
        print(f"Valid anxiety windows found (>=4 days): {len(self.validAnxietyWindow)}")
        print(f"Small anxiety windows found (3 days): {len(self.smallAnxietyWindow)}")
        print(f"Invalid anxiety windows found (2 days): {len(self.invalidAnxietyEntries)}")
        
        return self.validMoodWindow, self.validAnxietyWindow
    def removeDay(self):
        daysRemoved = input("Enter the amount of days you want to remove (1 or 2): ")
        if daysRemoved not in ['1', '2']:
            print("Invalid input. Please enter 1 or 2.")
            return None, None, None, None, None, None
        
        daysRemoved = int(daysRemoved)
        
        # Process valid mood data (>=4 entries, can remove 1 or 2)
        processedMoodWindow = []
        for i, entry in enumerate(self.validMoodWindow):
            try:
                parsed = [float(x.strip()) for x in entry.split(',')]
                for _ in range(daysRemoved):
                    if len(parsed) > 0:
                        indexRemove = random.randint(0, len(parsed) - 1)
                        parsed.pop(indexRemove)
                processedMoodWindow.append(parsed)
            except Exception as e:
                print(f"Error processing mood entry {i}: {e}")
                continue

        # Process small mood data (3 entries, can only remove 1)
        smallProcessedMoodWindow = []
        if daysRemoved == 1:
            for i, entry in enumerate(self.smallMoodWindow):
                try:
                    parsed = [float(x.strip()) for x in entry.split(',')]
                    if len(parsed) > 0:
                        indexRemove = random.randint(0, len(parsed) - 1)
                        parsed.pop(indexRemove)
                    smallProcessedMoodWindow.append(parsed)
                except Exception as e:
                    print(f"Error processing small mood entry {i}: {e}")
                    continue
        elif daysRemoved == 2:
            # Can't remove 2 from 3 entries, include them without removal
            print(f"Including {len(self.smallMoodWindow)} mood entries with only 3 days (no removal)")
            for i, entry in enumerate(self.smallMoodWindow):
                try:
                    parsed = [float(x.strip()) for x in entry.split(',')]
                    smallProcessedMoodWindow.append(parsed)
                except Exception as e:
                    print(f"Error processing small mood entry {i}: {e}")
                    continue

        # Process valid anxiety data (>=4 entries, can remove 1 or 2)
        processedAnxietyWindow = []
        for i, entry in enumerate(self.validAnxietyWindow):
            try:
                parsed = [float(x.strip()) for x in entry.split(',')]
                for _ in range(daysRemoved):
                    if len(parsed) > 0:
                        indexRemove = random.randint(0, len(parsed) - 1)
                        parsed.pop(indexRemove)
                processedAnxietyWindow.append(parsed)
            except Exception as e:
                print(f"Error processing anxiety entry {i}: {e}")
                continue

        # Process small anxiety data (3 entries, can only remove 1)
        smallProcessedAnxietyWindow = []
        if daysRemoved == 1:
            for i, entry in enumerate(self.smallAnxietyWindow):
                try:
                    parsed = [float(x.strip()) for x in entry.split(',')]
                    if len(parsed) > 0:
                        indexRemove = random.randint(0, len(parsed) - 1)
                        parsed.pop(indexRemove)
                    smallProcessedAnxietyWindow.append(parsed)
                except Exception as e:
                    print(f"Error processing small anxiety entry {i}: {e}")
                    continue
        elif daysRemoved == 2:
            # Can't remove 2 from 3 entries, include them without removal
            print(f"Including {len(self.smallAnxietyWindow)} anxiety entries with only 3 days (no removal)")
            for i, entry in enumerate(self.smallAnxietyWindow):
                try:
                    parsed = [float(x.strip()) for x in entry.split(',')]
                    smallProcessedAnxietyWindow.append(parsed)
                except Exception as e:
                    print(f"Error processing small anxiety entry {i}: {e}")
                    continue
            
        # Process invalid mood data (2 entries, cannot remove any)
        invalidMoodWindow = []
        for i, entry in enumerate(self.invalidMoodEntries):
            try:
                parsed = [float(x.strip()) for x in entry.split(',')]
                invalidMoodWindow.append(parsed)
            except Exception as e:
                print(f"Error processing invalid mood entry {i}: {e}")
                continue
        
        # Process invalid anxiety data (2 entries, cannot remove any)
        invalidAnxietyWindow = []
        for i, entry in enumerate(self.invalidAnxietyEntries):
            try:
                parsed = [float(x.strip()) for x in entry.split(',')]
                invalidAnxietyWindow.append(parsed)
            except Exception as e:
                print(f"Error processing invalid anxiety entry {i}: {e}")
                continue
        
        print(f"\nProcessed {len(processedMoodWindow)} valid mood entries")
        print(f"Processed {len(processedAnxietyWindow)} valid anxiety entries")
        print(f"Processed {len(smallProcessedMoodWindow)} small mood entries")
        print(f"Processed {len(smallProcessedAnxietyWindow)} small anxiety entries")
        print(f"Included {len(invalidMoodWindow)} invalid mood entries (2 days, no removal)")
        print(f"Included {len(invalidAnxietyWindow)} invalid anxiety entries (2 days, no removal)")
        
        return (processedMoodWindow, processedAnxietyWindow, 
                smallProcessedMoodWindow, smallProcessedAnxietyWindow, 
                invalidMoodWindow, invalidAnxietyWindow)
    
    def calculateMeanAndStdev(self, processedMoodWindow, processedAnxietyWindow,
                               smallProcessedMoodWindow, smallProcessedAnxietyWindow, 
                               invalidMoodWindow, invalidAnxietyWindow):
        """Calculate mean and standard deviation for each list"""
        results = []
        
        # Combine all lists to get total entries
        all_mood = processedMoodWindow + smallProcessedMoodWindow + invalidMoodWindow
        all_anxiety = processedAnxietyWindow + smallProcessedAnxietyWindow + invalidAnxietyWindow
        max_entries = max(len(all_mood), len(all_anxiety))
        
        # Process all mood entries
        for i in range(len(all_mood)):
            mood_values = all_mood[i]
            result_row = {
                'entry_id': i,
                'mood_values': str(mood_values) if mood_values else None,
                'mood_mean': np.mean(mood_values) if len(mood_values) > 0 else None,
                'mood_stdev': np.std(mood_values, ddof=1) if len(mood_values) > 1 else 0,
                'mood_count': len(mood_values) if mood_values else None,
                'anxiety_values': None,
                'anxiety_mean': None,
                'anxiety_stdev': None,
                'anxiety_count': None
            }
            
            # Add corresponding anxiety if available
            if i < len(all_anxiety):
                anxiety_values = all_anxiety[i]
                if len(anxiety_values) > 0:
                    result_row['anxiety_values'] = str(anxiety_values)
                    result_row['anxiety_mean'] = np.mean(anxiety_values)
                    result_row['anxiety_stdev'] = np.std(anxiety_values, ddof=1) if len(anxiety_values) > 1 else 0
                    result_row['anxiety_count'] = len(anxiety_values)
            
            results.append(result_row)
        
        # If there are more anxiety entries than mood entries
        for i in range(len(all_mood), len(all_anxiety)):
            anxiety_values = all_anxiety[i]
            result_row = {
                'entry_id': i,
                'mood_values': None,
                'mood_mean': None,
                'mood_stdev': None,
                'mood_count': None,
                'anxiety_values': str(anxiety_values) if anxiety_values else None,
                'anxiety_mean': np.mean(anxiety_values) if len(anxiety_values) > 0 else None,
                'anxiety_stdev': np.std(anxiety_values, ddof=1) if len(anxiety_values) > 1 else 0,
                'anxiety_count': len(anxiety_values) if anxiety_values else None
            }
            results.append(result_row)
        
        return results
    
    def saveResults(self, results, filename='results_output.csv'):
        """Save results to a CSV file"""
        df = pd.DataFrame(results)
        df.to_csv(filename, index=False)
        print(f"Results saved to {filename}")
        return df
    def run(self):
        """Main execution function"""
        print("=" * 60)
        print("Research Week 1 Data Analysis")
        print("=" * 60)
            
            # Load data
        print("\n1. Loading data...")
        self.loadData()
            
            # Parse data
        print("\n2. Parsing data for valid windows...")
        self.dataparse()
            
            # Remove days
        print("\n3. Removing random days...")
        (processedMood, processedAnxiety, 
         smallMood, smallAnxiety, 
         invalidMood, invalidAnxiety) = self.removeDay()
            
        if processedMood is None or processedAnxiety is None:
            print("Processing terminated due to error.")
            return
            
            # Calculate statistics
        print("\n4. Calculating mean and standard deviation...")
        results = self.calculateMeanAndStdev(processedMood, processedAnxiety, smallMood, smallAnxiety, invalidMood, invalidAnxiety)
            
            # Save results
        print("\n5. Saving results...")
        results_df = self.saveResults(results)
            
        print("\n" + "=" * 60)
        print("Processing complete!")
        print("=" * 60)
        print("\nSummary Statistics:")
        print(f"Total entries processed: {len(results)}")
        print(f"Mood entries: {results_df['mood_mean'].notna().sum()}")
        print(f"Anxiety entries: {results_df['anxiety_mean'].notna().sum()}")
            
        return results_df


if __name__ == "__main__":
        
    run_tool = Dataops()
    run_tool.run()