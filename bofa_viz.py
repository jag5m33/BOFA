import matplotlib.pyplot as plt

# Check if GH_CONTROL (Doped) scores are actually higher than ATHLETE_REF (Clean)
plt.figure(figsize=(10,6))
plt.hist(df[df['source'] == 'ATHLETE_REF']['total_suspicion'], bins=50, alpha=0.5, label='Clean')
plt.hist(df[df['source'] == 'GH_CONTROL']['total_suspicion'], bins=50, alpha=0.5, label='Doped')
plt.title("Suspicion Score Distribution")
plt.legend()
plt.show()