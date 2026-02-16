#!/bin/bash

# Improved Kafka + Dashboard Setup with proper waiting and error handling

set -e

echo "========================================"
echo "   Kafka + Dashboard Setup for Ubuntu"
echo "========================================"
echo ""

# Check if Java 17+ is available; upgrade if necessary for Kafka UI compatibility
REQUIRED_JAVA_MAJOR=17
if command -v java &> /dev/null; then
    JAVA_MAJOR=$(java -version 2>&1 | awk -F\" '/version/ {split($2, v, "."); if (v[1] == 1) {print v[2]} else {print v[1]}}')
    if [ -z "$JAVA_MAJOR" ]; then
        echo "Unable to determine installed Java version; forcing reinstall of OpenJDK ${REQUIRED_JAVA_MAJOR}."
        JAVA_MAJOR=0
    fi

    if [ "$JAVA_MAJOR" -lt "$REQUIRED_JAVA_MAJOR" ]; then
        echo "Java version $JAVA_MAJOR detected; upgrading to OpenJDK ${REQUIRED_JAVA_MAJOR}..."
        sudo apt update
        sudo apt install -y openjdk-17-jdk
    else
        echo "✓ Java $(java -version 2>&1 | head -n1)"
    fi
else
    echo "Installing OpenJDK ${REQUIRED_JAVA_MAJOR}..."
    sudo apt update
    sudo apt install -y openjdk-17-jdk
fi

# Set Kafka version
KAFKA_VERSION="3.6.1"
SCALA_VERSION="2.13"
KAFKA_DIR="$HOME/kafka_${SCALA_VERSION}-${KAFKA_VERSION}"
KAFKA_TAR="kafka_${SCALA_VERSION}-${KAFKA_VERSION}.tgz"

# Download and extract Kafka if not already present
if [ ! -d "$KAFKA_DIR" ]; then
    echo ""
    echo "Downloading Kafka ${KAFKA_VERSION}..."
    cd $HOME
    wget -q --show-progress https://archive.apache.org/dist/kafka/${KAFKA_VERSION}/${KAFKA_TAR}
    
    echo "Extracting Kafka..."
    tar -xzf ${KAFKA_TAR}
    rm ${KAFKA_TAR}
    
    echo "✓ Kafka extracted to $KAFKA_DIR"
    
    # Generate cluster ID and format storage
    cd $KAFKA_DIR
    CLUSTER_ID=$(bin/kafka-storage.sh random-uuid)
    echo "✓ Generated Cluster ID: $CLUSTER_ID"
    
    echo "Formatting storage..."
    bin/kafka-storage.sh format -t $CLUSTER_ID -c config/kraft/server.properties
    echo "✓ Storage formatted"
else
    echo "✓ Kafka already installed at $KAFKA_DIR"
fi

# Download Kafka UI if not present
KAFKA_UI_DIR="$HOME/kafka-ui"
KAFKA_UI_JAR="$KAFKA_UI_DIR/kafka-ui.jar"

if [ ! -f "$KAFKA_UI_JAR" ]; then
    echo ""
    echo "Downloading Kafka UI (Dashboard)..."
    mkdir -p $KAFKA_UI_DIR
    cd $KAFKA_UI_DIR
    wget -q --show-progress -O kafka-ui.jar \
        https://github.com/provectus/kafka-ui/releases/download/v0.7.2/kafka-ui-api-v0.7.2.jar
    echo "✓ Kafka UI downloaded"
else
    echo "✓ Kafka UI already installed"
fi

echo ""
echo "========================================"
echo "   Starting Kafka Server"
echo "========================================"
echo ""

# Kill any existing Kafka processes
pkill -f "kafka.Kafka" 2>/dev/null || true
sleep 2

# Start Kafka in background
cd $KAFKA_DIR
nohup bin/kafka-server-start.sh config/kraft/server.properties > $HOME/kafka.log 2>&1 &
KAFKA_PID=$!
echo "$KAFKA_PID" > $HOME/kafka.pid

echo "✓ Kafka starting (PID: $KAFKA_PID)"
echo "  Log file: $HOME/kafka.log"

# Wait for Kafka to be ready (check port 9092)
echo ""
echo "Waiting for Kafka to start (checking port 9092)..."
COUNTER=0
MAX_WAIT=30
while [ $COUNTER -lt $MAX_WAIT ]; do
    if nc -z localhost 9092 2>/dev/null; then
        echo "✓ Kafka is ready on localhost:9092"
        break
    fi
    echo -n "."
    sleep 1
    COUNTER=$((COUNTER + 1))
done

if [ $COUNTER -eq $MAX_WAIT ]; then
    echo ""
    echo "✗ Kafka failed to start within ${MAX_WAIT} seconds"
    echo "Check the log: tail -50 $HOME/kafka.log"
    exit 1
fi

echo ""
echo "========================================"
echo "   Starting Kafka UI Dashboard"
echo "========================================"
echo ""

# Kill any existing Kafka UI processes
pkill -f "kafka-ui" 2>/dev/null || true
sleep 2

# Start Kafka UI
cd $KAFKA_UI_DIR
export KAFKA_CLUSTERS_0_NAME=local
export KAFKA_CLUSTERS_0_BOOTSTRAPSERVERS=localhost:9092

nohup java -jar kafka-ui.jar --server.port=8080 > $HOME/kafka-ui.log 2>&1 &
KAFKA_UI_PID=$!
echo "$KAFKA_UI_PID" > $HOME/kafka-ui.pid

echo "✓ Kafka UI starting (PID: $KAFKA_UI_PID)"
echo "  Log file: $HOME/kafka-ui.log"

# Wait for Kafka UI to start (check port 8080)
echo ""
echo "Waiting for Kafka UI to start (checking port 8080)..."
COUNTER=0
MAX_WAIT=60
while [ $COUNTER -lt $MAX_WAIT ]; do
    if nc -z localhost 8080 2>/dev/null; then
        echo "✓ Kafka UI is ready on localhost:8080"
        break
    fi
    echo -n "."
    sleep 1
    COUNTER=$((COUNTER + 1))
done

if [ $COUNTER -eq $MAX_WAIT ]; then
    echo ""
    echo "✗ Kafka UI failed to start within ${MAX_WAIT} seconds"
    echo "Check the log: tail -50 $HOME/kafka-ui.log"
    exit 1
fi

echo ""
echo "========================================"
echo "   ✓ Setup Complete!"
echo "========================================"
echo ""
echo "Kafka Broker:    localhost:9092"
echo "Kafka Dashboard: http://localhost:8080"
echo ""
echo "Process IDs:"
echo "  - Kafka:    $KAFKA_PID"
echo "  - Dashboard: $KAFKA_UI_PID"
echo ""
echo "Logs:"
echo "  - Kafka:     tail -f $HOME/kafka.log"
echo "  - Dashboard: tail -f $HOME/kafka-ui.log"
echo ""
echo "To stop everything:"
echo "  ./kafka-stop.sh"
echo ""
echo "================================================"
echo "Open your browser: http://localhost:8080"
echo "================================================"
echo ""
