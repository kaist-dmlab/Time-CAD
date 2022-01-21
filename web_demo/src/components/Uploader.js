/* eslint-disable import/no-anonymous-default-export */
import React from 'react'
import { PageHeader, Upload, message } from 'antd';
import { FileAddFilled } from '@ant-design/icons';

const { Dragger } = Upload;

const props = {
    name: 'file',
    multiple: false,
    action: 'https://www.mocky.io/v2/5cc8019d300000980a055e76',
    onChange(info) {
        const { status } = info.file;
        if (status !== 'uploading') {
            console.log(info.file, info.fileList);
        }
        if (status === 'done') {
            message.success(`${info.file.name} file uploaded successfully.`);
        } else if (status === 'error') {
            message.error(`${info.file.name} file upload failed.`);
        }
    },
    onDrop(e) {
        console.log('Dropped files', e.dataTransfer.files);
    },
};

export default () => (
    <React.Fragment>
        <PageHeader title="File Uploader" />
        <Dragger {...props}>
            <p className="ant-upload-drag-icon">
                <FileAddFilled />
            </p>
            <p className="ant-upload-text">Click or drag file to this area to upload</p>
            <p className="ant-upload-hint">detailed description here... allow file extension ?</p>
        </Dragger>
    </React.Fragment>
)