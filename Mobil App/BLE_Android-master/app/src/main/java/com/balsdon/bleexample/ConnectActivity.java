package com.balsdon.bleexample;

import android.Manifest;
import android.app.AlertDialog;
import android.app.admin.DevicePolicyManager;
import android.bluetooth.BluetoothAdapter;
import android.content.ComponentName;
import android.content.Context;
import android.content.DialogInterface;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.os.Bundle;
import android.os.Handler;
import android.os.RemoteException;
import android.support.v4.app.ActivityCompat;
import android.support.v4.content.ContextCompat;
import android.support.v7.app.AppCompatActivity;
import android.text.method.ScrollingMovementMethod;
import android.util.Log;
import android.view.Menu;
import android.view.MenuItem;
import android.widget.TextView;
import android.widget.Toast;

import com.android.internal.telephony.ITelephony;

public class ConnectActivity extends AppCompatActivity implements BLEManager {

    private static final int MY_PERMISSIONS_REQUEST_ACCESS_FINE_LOCATION = 1;
    private static int REQUEST_ENABLE_BT = 10001;
    private static final String DEVICE_ID = "4afb720a-5214-4337-841b-d5f954214877";
    private static final String CHARACTERISTIC_ID = "8bacc104-15eb-4b37-bea6-0df3ac364199";     //"53b3d959-7dd3-4839-94e1-7b0eaea9aac2";

    private TextView mLogger;
    private String str = "";

    private static final String TAG = "Lockkkkk_Bluetooth";

    private DevicePolicyManager devicePolicyManager;
    private BLEPeripheral mPeripheral;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_connect);

        devicePolicyManager = (DevicePolicyManager) getSystemService(DEVICE_POLICY_SERVICE);


        if (ContextCompat.checkSelfPermission(ConnectActivity.this, Manifest.permission.READ_PHONE_STATE) != PackageManager.PERMISSION_GRANTED) {
            if (ActivityCompat.shouldShowRequestPermissionRationale(ConnectActivity.this, Manifest.permission.READ_PHONE_STATE)) {
                ActivityCompat.requestPermissions(ConnectActivity.this, new String[]{Manifest.permission.READ_PHONE_STATE}, 1);
            } else {
                ActivityCompat.requestPermissions(ConnectActivity.this, new String[]{Manifest.permission.READ_PHONE_STATE}, 1);
            }
        }

        if (!getPackageManager().hasSystemFeature(PackageManager.FEATURE_BLUETOOTH_LE)) {
            Toast.makeText(this, R.string.ble_not_supported, Toast.LENGTH_SHORT).show();
            finish();
        }

        mLogger = findViewById(R.id.log);
        mLogger.setMovementMethod(new ScrollingMovementMethod());

        mPeripheral = new BLEPeripheral(this, DEVICE_ID);

        TextView t = findViewById(R.id.information);
        t.setText(R.string.app_intro);
    }

    @Override
    public void log(final String message) {
        runOnUiThread(new Runnable() {
            @Override
            public void run() {
                Log.d(TAG, message);
                mLogger.setText(String.format("%s\n%s", message, mLogger.getText().toString()));
            }
        });
    }

    @Override
    public void onConnectionStateChange(int newState) {

    }


    @Override
    public Context getContext() {
        return this;
    }

    @Override
    public void enableBluetooth() {
        Intent enableBtIntent = new Intent(BluetoothAdapter.ACTION_REQUEST_ENABLE);
        startActivityForResult(enableBtIntent, REQUEST_ENABLE_BT);
    }

    @Override
    public void onConnected() {
        runOnUiThread(new Runnable() {
            IncomingCallReceiver income;

            @Override
            public void run() {
                final Handler handler = new Handler();
                Runnable runnable = new Runnable() {
                    @Override
                    public void run() {
                        income = new IncomingCallReceiver();
                        income.onReceive(getApplicationContext(), getIntent());
                        //Log.d(TAG + " Sear", "hhhhh");
                        mPeripheral.readCharacteristic(CHARACTERISTIC_ID);

                        str = mPeripheral.getString();
                        Log.d(TAG + " str ", str);
                        if (str.contains("Lock")) {
                            Log.d(TAG + "lock", ": ");
                            setLock();
                        } else if (str.contains("Calling")) {
                            Log.d(TAG, "run: calling coming");
                            ITelephony telephonyService = income.getTelephonyService();
                            if (null != telephonyService) {
                                Log.d(TAG, "calling not null");
                                try {
                                    telephonyService.silenceRinger();
                                    telephonyService.endCall();
                                    Toast.makeText(getApplicationContext(), "you missed a call \nPark and call back ", Toast.LENGTH_SHORT).show();
                                } catch (RemoteException e) {
                                    e.printStackTrace();
                                }
                            }
                        }
                        handler.postDelayed(this, 500);
                    }

                };
                handler.post(runnable);
            }
        });

        mPeripheral.subscribe(CHARACTERISTIC_ID, valueChanged);

    }

//    private void displayDialog() {
//        AlertDialog.Builder alertDialogBuilder = new AlertDialog.Builder(
//                ConnectActivity.this);
//
//        // set title
//        alertDialogBuilder.setTitle("Call Rejected!!!");
//
//        // set dialog message
//        alertDialogBuilder
//                .setMessage("Better to stop and call him back!!")
//                .setPositiveButton("Ok", new DialogInterface.OnClickListener() {
//                    public void onClick(DialogInterface dialog, int id) {
//                        dialog.dismiss();
//                    }
//                });
//
//        // create alert dialog
//        AlertDialog alertDialog = alertDialogBuilder.create();
//
//        // show it
//        alertDialog.show();
//    }

    private Command<String> valueChanged = new Command<String>() {
        @Override
        public void execute(String data) {
            log("Value result == " + data);
        }
    };

    @Override
    public void onDisconnected() {
        str = "Safe";
        runOnUiThread(new Runnable() {
            @Override
            public void run() {
                //str = "Safe";
//                mSendData.setEnabled(false);
//                mReadButton.setEnabled(false);
            }
        });
    }

    public void setLock() {
        devicePolicyManager.lockNow();
    }

    @Override
    public boolean checkPermission() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.ACCESS_FINE_LOCATION) != PackageManager.PERMISSION_GRANTED) {

            // Should we show an explanation?
            if (ActivityCompat.shouldShowRequestPermissionRationale(this, Manifest.permission.ACCESS_FINE_LOCATION)) {
                // Show an explanation to the user *asynchronously* -- don't block
                // this thread waiting for the user's response! After the user
                // sees the explanation, try again to request the permission.
            } else {
                ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.ACCESS_FINE_LOCATION}, MY_PERMISSIONS_REQUEST_ACCESS_FINE_LOCATION);
            }
            return false;
        }

        return true;
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, String permissions[], int[] grantResults) {
        switch (requestCode) {
            case MY_PERMISSIONS_REQUEST_ACCESS_FINE_LOCATION: {
                if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                    mPeripheral.scanForDevice();
                    if (ContextCompat.checkSelfPermission(ConnectActivity.this, Manifest.permission.READ_PHONE_STATE) == PackageManager.PERMISSION_GRANTED) {
                        Toast.makeText(this, "Permiss", Toast.LENGTH_SHORT).show();
                    }
                } else {
                    Toast.makeText(this, "Permission denied. App is a brick !!", Toast.LENGTH_SHORT).show();
                    log("Permission denied. App is a brick");
                }
                return;
            }

            // other 'case' lines to check for other
            // permissions this app might request
        }
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        getMenuInflater().inflate(R.menu.bluetooth_chat, menu);
        return true;
    }


    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        switch (item.getItemId()) {

            case R.id.permission: {
                // Launch the DeviceListActivity to see devices and do scan
                //Intent serverIntent = new Intent(getApplicationContext(), DeviceListActivity.class);
                //startActivityForResult(serverIntent, REQUEST_CONNECT_DEVICE_INSECURE);
                startActivity(new Intent(getApplicationContext(), LockScreen.class));
                return true;
            }
        }
        return false;
    }

}
