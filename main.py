import functions as fnc
import xarray as xr


if __name__ == '__main__':
    # Read the sample input NetCDF file
    data = xr.open_dataset('Data/Sample Data MT8.NC', mask_and_scale=True)

    # Compile the RGB composite image
    fnc.RGBmap(data)

    # Extract the T-re profile
    d15_d70 = fnc.mask(data)

    # Run the logistic regression based nowcsasting
    fnc.Logistic(d15_d70)
