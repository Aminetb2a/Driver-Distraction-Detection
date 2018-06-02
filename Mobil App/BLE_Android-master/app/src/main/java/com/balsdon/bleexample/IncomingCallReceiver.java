package com.balsdon.bleexample;

/**
 * Created by HilalKazan on 10.01.2018.
 */

import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.media.AudioManager;
import android.os.Handler;
import android.os.RemoteException;
import android.telephony.TelephonyManager;
import android.util.Log;
import android.view.KeyEvent;
import android.widget.Toast;

import com.android.internal.telephony.ITelephony;

import java.lang.reflect.Method;
import java.util.Objects;


public class IncomingCallReceiver extends BroadcastReceiver {
    String incomingNumber = "";
    AudioManager audioManager;
    TelephonyManager telephonyManager;
    private ITelephony telephonyService;
    private String blacklistednumber = "";
    private BLEPeripheral mPeripheral;
    private static final String CHARACTERISTIC_ID = "8bacc104-15eb-4b37-bea6-0df3ac364199";//"53b3d959-7dd3-4839-94e1-7b0eaea9aac2";
    private String str = "";

    private static final String TAG = "Lockkkkk_CALL";

    //    public IncomingCallReceiver(Context context, Intent intent, TelephonyManager telephonyManager, ITelephony telephonyService, String blacklistednumber, BLEPeripheral mPeripheral) {
//        this.telephonyManager = telephonyManager;
//        this.telephonyService = telephonyService;
//        this.blacklistednumber = blacklistednumber;
//        this.mPeripheral = mPeripheral;
//        onReceive(context, intent);
//    }

    public ITelephony getTelephonyService() {
        return telephonyService;
    }

    public void onReceive(Context context, Intent intent) {
        try {
           telephonyManager = (TelephonyManager) context.getSystemService(Context.TELEPHONY_SERVICE);
            Class clazz = Class.forName(telephonyManager.getClass().getName());
            Method method = clazz.getDeclaredMethod("getITelephony");
            method.setAccessible(true);
            telephonyService = (ITelephony) method.invoke(telephonyManager);

            String state = intent.getStringExtra(TelephonyManager.EXTRA_STATE);
            blacklistednumber = mPeripheral.getString();
            //blacklistednumber = connectActivity.getSign();
            Log.d("str !=!", blacklistednumber);
            if (state.equals(TelephonyManager.EXTRA_STATE_RINGING)) {
                incomingNumber = intent.getStringExtra(TelephonyManager.EXTRA_INCOMING_NUMBER);
                Log.d(TAG + "onReceive ", incomingNumber);
                Toast.makeText(context, incomingNumber, Toast.LENGTH_SHORT).show();
                //telephonyService.endCall();
                if (blacklistednumber.contains("Calling")) {
                    Log.d(TAG + "INN ==: ", blacklistednumber);
                    telephonyService.silenceRinger();
                    telephonyService.endCall();
                    Log.e(TAG + "HANG UP", incomingNumber);
                }

            }
            if (state.equals(TelephonyManager.EXTRA_STATE_OFFHOOK)) {
                Toast.makeText(context, TAG + "Received", Toast.LENGTH_SHORT).show();

            }
            if (state.equals(TelephonyManager.EXTRA_STATE_IDLE)) {
                Toast.makeText(context, "idle", Toast.LENGTH_SHORT).show();
            }


            final Handler handler = new Handler();
            Runnable runnable = new Runnable() {
                @Override
                public void run() {
                    Log.d(TAG + "Searching ", "hhhhh");
                    mPeripheral.readCharacteristic(CHARACTERISTIC_ID);

                    str = mPeripheral.getString();
                    Log.d(TAG + "str === ", str);
                    if (blacklistednumber.contains("Calling")) {
                        Log.d(TAG + "INN ==: ", blacklistednumber);
                        try {
                            telephonyService.silenceRinger();
                            telephonyService.endCall();
                        } catch (RemoteException e) {
                            e.printStackTrace();
                        }

                        Log.e("HANG UP", incomingNumber);
                    }
                    handler.postDelayed(this, 500);
                }

            };
            handler.post(runnable);


//            if ((incomingNumber != null) && incomingNumber.equals(blacklistednumber)) {
//                Log.d( "INN ==: ", incomingNumber);
//                telephonyService.silenceRinger();
//                telephonyService.endCall();
//                Log.e("HANG UP", incomingNumber);
//            }

            //ConnectActivity connectActivity = new ConnectActivity();

        } catch (Exception e) {
            e.printStackTrace();
        }

        // Get AudioManager
        audioManager = (AudioManager) context.getSystemService(Context.AUDIO_SERVICE);
        // Get TelephonyManager
        telephonyManager = (TelephonyManager) context.getSystemService(Context.TELEPHONY_SERVICE);
        if (Objects.equals(intent.getAction(), "android.intent.action.PHONE_STATE")) {
            String state = intent.getStringExtra(TelephonyManager.EXTRA_STATE);
            Log.d(state, "state ==: ");
            if (state.equals(TelephonyManager.EXTRA_STATE_RINGING)) {
                // Get incoming number
                incomingNumber = intent.getStringExtra(TelephonyManager.EXTRA_INCOMING_NUMBER);
                Log.d("onReceive ==: ", incomingNumber);

            }
        }

    }


    private void startApp(Context context, String number) {
        Intent intent = new Intent(context, ConnectActivity.class);
        intent.setFlags(Intent.FLAG_ACTIVITY_NEW_TASK);
        intent.putExtra("number", "Rejected incoming number:" + number);
        context.startActivity(intent);
    }

    private void rejectCall(Context context) {
        Intent buttonDown = new Intent(Intent.ACTION_MEDIA_BUTTON);
        buttonDown.putExtra(Intent.EXTRA_KEY_EVENT,
                new KeyEvent(KeyEvent.ACTION_DOWN, KeyEvent.KEYCODE_HEADSETHOOK));
        context.sendOrderedBroadcast(buttonDown, "android.permission.CALL_PRIVILEGED");
    }

    private void rejectCall() {


        try {

            // Get the getITelephony() method
            Class<?> classTelephony = Class.forName(telephonyManager.getClass().getName());
            Method method = classTelephony.getDeclaredMethod("getITelephony");
            // Disable access check
            method.setAccessible(true);
            // Invoke getITelephony() to get the ITelephony interface
            Object telephonyInterface = method.invoke(telephonyManager);
            // Get the endCall method from ITelephony
            Class<?> telephonyInterfaceClass = Class.forName(telephonyInterface.getClass().getName());
            Method methodEndCall = telephonyInterfaceClass.getDeclaredMethod("endCall");
            // Invoke endCall()
            methodEndCall.invoke(telephonyInterface);
        } catch (Exception e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }

    }
}