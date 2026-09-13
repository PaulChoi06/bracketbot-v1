let 
pkgs = import (fetchTarball {
  url = "https://github.com/NixOS/nixpkgs/archive/63dacb46bf939521bdc93981b4cbb7ecb58427a0.tar.gz";
  sha256 = "sha256:1lr1h35prqkd1mkmzriwlpvxcb34kmhc9dnr48gkm8hh089hifmx";
}) {};
in
let 
hotspot = pkgs.writeShellApplication {
  name = "hotspot";
  runtimeInputs = [ pkgs.dnsmasq ];
  text = ''
    # initialize hotspot
    host=$(hostname)
    SSID="$host"
    PASSWORD="12345678"
    CONNECTION_NAME="Hotspot"
    WIFI_IFACE="wlP1p1s0"

configure_hotspot() {
    echo "Configuring NetworkManager hotspot on $WIFI_IFACE..."
    if nmcli c show "$CONNECTION_NAME" > /dev/null 2>&1; then
        echo "Deleting existing connection '$CONNECTION_NAME'..."
        sudo nmcli connection delete "$CONNECTION_NAME" || echo "Failed to delete existing connection $CONNECTION_NAME"
        sleep 1
    fi
    echo "Creating new hotspot connection '$CONNECTION_NAME'..."
    sudo nmcli connection add type wifi ifname "$WIFI_IFACE" con-name "$CONNECTION_NAME" autoconnect yes ssid "$SSID" || echo "Failed to add connection $CONNECTION_NAME"
    sleep 1
    sudo nmcli connection modify "$CONNECTION_NAME" 802-11-wireless.mode ap || echo "Failed: set mode ap"
    sudo nmcli connection modify "$CONNECTION_NAME" 802-11-wireless.band bg || echo "Failed: set band bg"
    sudo nmcli connection modify "$CONNECTION_NAME" ipv4.method shared || echo "Failed: set ipv4 shared"
    sudo nmcli connection modify "$CONNECTION_NAME" wifi-sec.key-mgmt wpa-psk || echo "Failed: set key-mgmt wpa-psk"
    sudo nmcli connection modify "$CONNECTION_NAME" wifi-sec.psk "$PASSWORD" || echo "Failed: set psk"
    sudo nmcli connection modify "$CONNECTION_NAME" connection.autoconnect-priority 999 || echo "Failed: set autoconnect-priority"
    sleep 1
    echo "Activating hotspot connection '$CONNECTION_NAME'..."
    sudo nmcli connection up "$CONNECTION_NAME" || echo "Failed to bring up connection $CONNECTION_NAME"
    sleep 1
}
configure_ssh() {
    echo "Ensuring SSH service is enabled and running..."
    if ! systemctl is-active ssh >/dev/null 2>&1; then
        sudo systemctl enable ssh || echo "Failed: enable ssh"
        sudo systemctl start ssh || echo "Failed: start ssh"
    else
        sudo systemctl restart ssh || echo "Failed: restart ssh"
    fi
}

configure_hotspot
configure_ssh

echo "Access Point Setup script finished."  

    
  '';
};
in pkgs.buildEnv {
  name = "hotspot";
  paths = [ hotspot ];
}
