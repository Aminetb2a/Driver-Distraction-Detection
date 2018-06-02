package com.balsdon.bleexample;

import android.app.admin.DeviceAdminReceiver;
import android.content.Context;
import android.content.Intent;
import android.widget.Toast;

/**
 * Created by HilalKazan on 09.01.2018.
 */


public class MyAdmin extends DeviceAdminReceiver {
    @Override
    public void onEnabled(Context context, Intent intent) {
        Toast.makeText(context,"Enabled",Toast.LENGTH_SHORT).show();
    }
    @Override
    public void onDisabled(Context context, Intent intent){
        Toast.makeText(context,"Disabled",Toast.LENGTH_SHORT).show();
    }

}