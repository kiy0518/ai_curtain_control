"""Inject BLE permissions into the freshly-generated AndroidManifest.xml.

Run after `flutter create .` (which regenerates the manifest). Idempotent.
"""
import re
import sys

MANIFEST = sys.argv[1] if len(sys.argv) > 1 \
    else "android/app/src/main/AndroidManifest.xml"

PERMS = """    <uses-permission android:name="android.permission.BLUETOOTH_SCAN" android:usesPermissionFlags="neverForLocation"/>
    <uses-permission android:name="android.permission.BLUETOOTH_CONNECT"/>
    <uses-permission android:name="android.permission.BLUETOOTH" android:maxSdkVersion="30"/>
    <uses-permission android:name="android.permission.BLUETOOTH_ADMIN" android:maxSdkVersion="30"/>
    <uses-permission android:name="android.permission.ACCESS_FINE_LOCATION" android:maxSdkVersion="30"/>
    <uses-feature android:name="android.hardware.bluetooth_le" android:required="true"/>
"""

s = open(MANIFEST).read()
if "BLUETOOTH_CONNECT" not in s:
    s = re.sub(r"(<manifest[^>]*>\s*\n)", r"\1" + PERMS, s, count=1)
    open(MANIFEST, "w").write(s)
    print("BLE permissions injected.")
else:
    print("permissions already present.")
