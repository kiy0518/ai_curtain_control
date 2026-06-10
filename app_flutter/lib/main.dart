// AI Curtain — BLE remote (Flutter, Material 3). Minimal UI:
//   BLE 찾기/연결 · 연결 해제(제거) · 열기/정지/닫기
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
  BluetoothCharacteristic? _cmd;
  String _status = "연결 안 됨";
  bool _busy = false;
  StreamSubscription? _scanSub, _connSub;

  bool get _connected => _device != null && _cmd != null;

  @override
  void dispose() {
    _scanSub?.cancel();
    _connSub?.cancel();
    _device?.disconnect();
    super.dispose();
  }

  // BLE 찾기 + 연결
  Future<void> _findAndConnect() async {
    setState(() { _busy = true; _status = "검색 중…"; });
    try {
      await FlutterBluePlus.startScan(
          withServices: [Guid(svcUuid)], timeout: const Duration(seconds: 8));
      _scanSub = FlutterBluePlus.scanResults.listen((results) async {
        if (results.isEmpty) return;
        await FlutterBluePlus.stopScan();
        await _scanSub?.cancel();
        await _connectTo(results.first.device);
      });
      Future.delayed(const Duration(seconds: 9), () {
        if (!_connected && mounted) {
          setState(() { _busy = false; if (_status == "검색 중…") _status = "기기 못 찾음"; });
        }
      });
    } catch (e) {
      setState(() { _busy = false; _status = "오류: $e"; });
    }
  }

  Future<void> _connectTo(BluetoothDevice dev) async {
    setState(() => _status = "연결 중…");
    _connSub = dev.connectionState.listen((s) {
      if (s == BluetoothConnectionState.disconnected && mounted) {
        setState(() { _device = null; _cmd = null; _status = "연결 끊김"; });
      }
    });
    await dev.connect(timeout: const Duration(seconds: 10));
    for (final s in await dev.discoverServices()) {
      if (s.uuid != Guid(svcUuid)) continue;
      for (final c in s.characteristics) {
        if (c.uuid == Guid(cmdUuid)) _cmd = c;
      }
    }
    setState(() { _device = dev; _busy = false; _status = "연결됨"; });
  }

  // 제거(연결 해제)
  Future<void> _disconnect() async {
    await _connSub?.cancel();
    await _device?.disconnect();
    setState(() { _device = null; _cmd = null; _status = "연결 안 됨"; });
  }

  Future<void> _send(String c) async {
    if (_cmd == null) return;
    await _cmd!.write(utf8.encode(c), withoutResponse: true);
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('🪟 AI 커튼 리모컨'),
        actions: [Padding(
          padding: const EdgeInsets.only(right: 14),
          child: Center(child: Text(_status, style: const TextStyle(fontSize: 13))))],
      ),
      body: Padding(
        padding: const EdgeInsets.all(24),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            if (!_connected) ...[
              FilledButton.icon(
                onPressed: _busy ? null : _findAndConnect,
                icon: const Icon(Icons.bluetooth_searching),
                label: Text(_busy ? "검색/연결 중…" : "BLE 찾기 · 연결"),
                style: FilledButton.styleFrom(minimumSize: const Size.fromHeight(64)),
              ),
            ] else ...[
              _ctlBtn("열기", Icons.keyboard_arrow_up, const Color(0xFF2E7D32), () => _send("OPEN")),
              const SizedBox(height: 14),
              _ctlBtn("정지", Icons.stop, const Color(0xFFEF6C00), () => _send("STOP")),
              const SizedBox(height: 14),
              _ctlBtn("닫기", Icons.keyboard_arrow_down, const Color(0xFFC62828), () => _send("CLOSE")),
              const SizedBox(height: 28),
              OutlinedButton.icon(
                onPressed: _disconnect,
                icon: const Icon(Icons.bluetooth_disabled),
                label: const Text("연결 해제 (제거)"),
                style: OutlinedButton.styleFrom(minimumSize: const Size.fromHeight(52)),
              ),
            ],
          ],
        ),
      ),
    );
  }

  Widget _ctlBtn(String label, IconData icon, Color color, VoidCallback onTap) {
    return FilledButton.icon(
      onPressed: onTap,
      icon: Icon(icon, size: 28),
      label: Text(label, style: const TextStyle(fontSize: 22, fontWeight: FontWeight.bold)),
      style: FilledButton.styleFrom(
          backgroundColor: color, minimumSize: const Size.fromHeight(84)),
    );
  }
}
