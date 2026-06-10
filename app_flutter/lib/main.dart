// AI Curtain — BLE remote control (Flutter, Material 3).
// Scans for the board's "AI-Curtain" Curtain Control service, connects, sends
// OPEN/CLOSE/STOP, and shows live status via notifications.
//
// Board GATT (src/ble_server.py):
//   Service c0de0000-1212-efde-1523-785feabcd123
//   Command c0de0001-... (write)   Status c0de0002-... (read/notify)

import 'dart:async';
import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:flutter_blue_plus/flutter_blue_plus.dart';

const svcUuid = "c0de0000-1212-efde-1523-785feabcd123";
const cmdUuid = "c0de0001-1212-efde-1523-785feabcd123";
const staUuid = "c0de0002-1212-efde-1523-785feabcd123";

void main() => runApp(const CurtainApp());

class CurtainApp extends StatelessWidget {
  const CurtainApp({super.key});
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'AI 커튼',
      theme: ThemeData(
        useMaterial3: true,
        colorScheme: ColorScheme.fromSeed(
            seedColor: const Color(0xFFD0BCFF), brightness: Brightness.dark),
      ),
      home: const RemotePage(),
    );
  }
}

class RemotePage extends StatefulWidget {
  const RemotePage({super.key});
  @override
  State<RemotePage> createState() => _RemotePageState();
}

class _RemotePageState extends State<RemotePage> {
  BluetoothDevice? _device;
  BluetoothCharacteristic? _cmd, _sta;
  String _status = "연결 안 됨";
  String _curtain = "—";
  bool _connecting = false;
  StreamSubscription? _scanSub, _staSub, _connSub;

  @override
  void dispose() {
    _scanSub?.cancel();
    _staSub?.cancel();
    _connSub?.cancel();
    _device?.disconnect();
    super.dispose();
  }

  Future<void> _connect() async {
    setState(() { _connecting = true; _status = "스캔 중…"; });
    await FlutterBluePlus.startScan(
        withServices: [Guid(svcUuid)], timeout: const Duration(seconds: 8));
    _scanSub = FlutterBluePlus.scanResults.listen((results) async {
      if (results.isEmpty) return;
      await FlutterBluePlus.stopScan();
      _scanSub?.cancel();
      final dev = results.first.device;
      setState(() { _status = "연결 중…"; });
      _connSub = dev.connectionState.listen((s) {
        if (s == BluetoothConnectionState.disconnected) {
          setState(() { _status = "연결 끊김"; _cmd = null; _sta = null; });
        }
      });
      await dev.connect(timeout: const Duration(seconds: 10));
      final services = await dev.discoverServices();
      for (final s in services) {
        if (s.uuid != Guid(svcUuid)) continue;
        for (final c in s.characteristics) {
          if (c.uuid == Guid(cmdUuid)) _cmd = c;
          if (c.uuid == Guid(staUuid)) {
            _sta = c;
            await c.setNotifyValue(true);
            _staSub = c.onValueReceived.listen(_onStatus);
          }
        }
      }
      setState(() { _device = dev; _status = "연결됨"; _connecting = false; });
    });
    // scan timeout fallback
    Future.delayed(const Duration(seconds: 9), () {
      if (_device == null && mounted) {
        setState(() { _connecting = false; if (_status == "스캔 중…") _status = "기기 못 찾음"; });
      }
    });
  }

  void _onStatus(List<int> v) {
    try {
      final m = jsonDecode(utf8.decode(v));
      const kr = {"OPEN": "열림", "CLOSED": "닫힘", "STOPPED": "정지", "UNKNOWN": "—"};
      setState(() => _curtain = kr[m["state"]] ?? "${m["state"]}");
    } catch (_) {}
  }

  Future<void> _send(String c) async {
    if (_cmd == null) return;
    await _cmd!.write(utf8.encode(c), withoutResponse: true);
  }

  @override
  Widget build(BuildContext context) {
    final connected = _device != null && _cmd != null;
    return Scaffold(
      appBar: AppBar(title: const Text('🪟 AI 커튼 리모컨'), actions: [
        Padding(padding: const EdgeInsets.only(right: 12),
            child: Center(child: Text(_status, style: const TextStyle(fontSize: 13)))),
      ]),
      body: Padding(
        padding: const EdgeInsets.all(20),
        child: Column(children: [
          Card(child: Padding(padding: const EdgeInsets.all(28),
            child: Column(children: [
              const Text("커튼 상태", style: TextStyle(fontSize: 14)),
              const SizedBox(height: 8),
              Text(_curtain, style: const TextStyle(fontSize: 40, fontWeight: FontWeight.bold)),
            ]))),
          const SizedBox(height: 24),
          if (!connected)
            FilledButton.icon(
              onPressed: _connecting ? null : _connect,
              icon: const Icon(Icons.bluetooth_searching),
              label: Text(_connecting ? "연결 중…" : "기기 연결"),
              style: FilledButton.styleFrom(minimumSize: const Size.fromHeight(56)),
            )
          else ...[
            _btn("열기", Icons.keyboard_arrow_up, const Color(0xFF2E7D32), () => _send("OPEN")),
            const SizedBox(height: 12),
            _btn("정지", Icons.stop, const Color(0xFFEF6C00), () => _send("STOP")),
            const SizedBox(height: 12),
            _btn("닫기", Icons.keyboard_arrow_down, const Color(0xFFC62828), () => _send("CLOSE")),
          ],
        ]),
      ),
    );
  }

  Widget _btn(String label, IconData icon, Color color, VoidCallback onTap) {
    return FilledButton.icon(
      onPressed: onTap,
      icon: Icon(icon),
      label: Text(label, style: const TextStyle(fontSize: 20)),
      style: FilledButton.styleFrom(
          backgroundColor: color, minimumSize: const Size.fromHeight(72)),
    );
  }
}
