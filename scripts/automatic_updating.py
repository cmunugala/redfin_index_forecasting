import os
import glob

if __name__ == '__main__':

    script_dir = os.path.dirname(os.path.realpath(__file__))
    print(script_dir)

    with open(f'{script_dir}/../README.md', 'r') as file:
        readme_lines = file.readlines()

    plots = glob.glob(f'{script_dir}/../forecast_plots/*forecast*')
    plots = sorted([x for x in plots if 'zoomed' not in x])
    zoomed_plots = sorted(glob.glob(f'{script_dir}/../forecast_plots/*zoomed*'))

    latest_zoomed = zoomed_plots[-1].split('../')[1]
    latest_normal = plots[-1].split('../')[1]

    for i, line in enumerate(readme_lines):
        if '![alt text]' in line:
            if 'zoomed' in line:
                new_line = f'![alt text]({latest_zoomed})\n'
                readme_lines[i] = new_line
            elif 'forecast.png' in line:
                new_line = f'![alt text]({latest_normal})\n'
                readme_lines[i] = new_line
            else:
                continue
    
    with open(f'{script_dir}/../README.md', 'w') as file:
        file.writelines(readme_lines)