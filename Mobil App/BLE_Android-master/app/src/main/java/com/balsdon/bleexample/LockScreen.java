package com.balsdon.bleexample;

/**
 * Created by HilalKazan on 09.01.2018.
 */

import android.app.Activity;
import android.app.ActivityManager;
import android.app.admin.DevicePolicyManager;
import android.content.ComponentName;
import android.content.Intent;
import android.graphics.Typeface;
import android.os.Bundle;
import android.support.annotation.Nullable;
import android.support.v7.app.AppCompatActivity;
import android.text.SpannableString;
import android.text.style.StyleSpan;
import android.view.View;
import android.widget.Button;
import android.widget.TextView;
import android.widget.Toast;

public class LockScreen extends AppCompatActivity {

    public static final int Result_Enable = 11;
    Button enable, disable;;
    Button goTo;
    private DevicePolicyManager devicePolicyManager;
    private ActivityManager activityManager;
    private ComponentName compName;



    @Override
    protected void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_lock);

        devicePolicyManager = (DevicePolicyManager) getSystemService(DEVICE_POLICY_SERVICE);
        activityManager = (ActivityManager) getSystemService(ACTIVITY_SERVICE);
        compName = new ComponentName(this, MyAdmin.class);

        TextView t = findViewById(R.id.textView);
        t.setText("          Please enable \n      to be able to access \nthe lock screen permission ");
//        SpannableString ss1=  new SpannableString(t);
//        ss1.setSpan(new StyleSpan(Typeface.BOLD), 0, ss1.length(), 0);


        enable = findViewById(R.id.enableBtn);
        disable = (Button) findViewById(R.id.disableBtn);
        goTo = findViewById(R.id.move_to_main);

        goTo.setVisibility(View.GONE);

        enable.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Intent intent = new Intent(DevicePolicyManager.ACTION_ADD_DEVICE_ADMIN);
                intent.putExtra(DevicePolicyManager.EXTRA_DEVICE_ADMIN, compName);
                intent.putExtra(DevicePolicyManager.EXTRA_ADD_EXPLANATION, "Additional text explaining why we need this permission");
                startActivityForResult(intent, Result_Enable);
            }
        });

        disable.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                devicePolicyManager.removeActiveAdmin(compName);
                disable.setVisibility(view.GONE);
                enable.setVisibility(view.VISIBLE);
                goTo.setVisibility(View.GONE);
            }
        });

        goTo.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                startActivity(new Intent(getApplicationContext(), ConnectActivity.class));
            }
        });


    }


    @Override
    protected void onResume() {
        super.onResume();
        boolean isActive = devicePolicyManager.isAdminActive(compName);
        disable.setVisibility(isActive ? View.VISIBLE : View.GONE );
        enable.setVisibility(isActive ? View.GONE : View.VISIBLE );
        goTo.setVisibility(isActive ? View.VISIBLE : View.GONE);
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        switch (requestCode) {
            case Result_Enable:
                if (resultCode == Activity.RESULT_OK) {
                    Toast.makeText(LockScreen.this, "you have enabled the Admin Device features", Toast.LENGTH_SHORT).show();
                    goTo.setVisibility(View.VISIBLE);
                } else {
                    Toast.makeText(LockScreen.this, "Problem to enabled the Admin Device features", Toast.LENGTH_SHORT).show();
                }
                break;
        }
        super.onActivityResult(requestCode, resultCode, data);
    }


    public void setLock() {

        boolean active = devicePolicyManager.isAdminActive(compName);

        if (active) {
            devicePolicyManager.lockNow();
        } else {
            Toast.makeText(this, "you need to enable Admin permissions", Toast.LENGTH_SHORT).show();
            Toast.makeText(this, "you need to enable Admin permissions", Toast.LENGTH_SHORT).show();
        }

    }
}
